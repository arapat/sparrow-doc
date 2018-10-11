# Sparrow Pseudocode

![System Design](https://www.lucidchart.com/publicSegments/view/e4b4d91a-c1ce-477a-a304-1492e341d8a4/image.png)

## Basic data types

### LabeledData

#### Structure

```
LabeledData := (Feature, Label)
```

* **Feature**: Represents a set of binary features compiled for a raw example.
* **Label**: Either +1 or -1 (for binary classification)

### SampledExample

Define an example in its compressed form for _training_.
This is the format of examples in BufferLoader (defined below).
The inner-loop of Sparrow iterates over tables with SampledExample as rows.

```
SampledExample := (LabeledData, SampledScore, SampledTreeIndex, LastScore, LastTreeIndex)
```

* **Score** is the weighted sum of the weak rule for this example
* **TreeIndex** is the index of the last weak rule added to a particular tree. The core is updated by adding to the old score the weighted sum of the rules added since then.
* There are two pairs of **Score,TreeIndex**:
   * `Sampled-` corresponds to time at which this example was sampled
   * `Last-` corresponds to the last time the weight of this example was updated.

### ScoredExample

Define an example in its compressed form for _sampling_.
This is the format of examples in StratifiedStorage (defined below).

```
ScoredExample := (LabeledData, LastScore, LastTreeIndex)
```

### Tree

#### Structure

```
Node := (SplitIndex, SplitThreshold, Prediction, LeftChildIndex, RightChildIndex)
Tree := Array[Node1, Node2, ...]
```

* `SplitIndex`: The feature index with which the tree node split examples into the left and right tree nodes.
* `SplitThreshold`: The threshold for splitting using the feature value.
* `Prediction`: The prediction of the tree nodes.
* `LeftChildIndex`, `RightChildIndex`: The indices of the left and right children of the tree nodes.

## Key Components and Operations

### Run Sparrow

There are three key components of Sparrow: the stratified storage, the buffered loader, and the learning algorithm. Their internal implementations are discussed later.

```
Procedure RunSparrow():
  StartStratifiedStorage()
  StartSampledExamplesGatherer()
  RunTraining()
END
```

We also need to define a function that computes the weight of an example given its label and its score given by the ensemble.

```
Procedure GetWeight(Label, Score):
  RETURN <the weight of the example, e.g., EXP(-Label * Score) for AdaBoost>
END
```


### BufferLoader

`BufferLoader` is an in-memory data loader to efficiently provide training examples to the learning algorithm. It maintains a weighted sample of the training dataset with regard to the latest ensemble model. In a separate thread, it constantly gather newly sampled examples from the stratified storage to replace its current sample set.

#### Structure

```
BufferLoader := (Size, BatchSize, LoadedData, LoadingData, ESS)
LoadedData   := CircularQueue[SampledExample1, SampledExample2, ...]
LoadingData  := <same as LoadedData>
```

* `Size`: The capacity of the BufferLoader
* `BatchSize`: The size of a batch when reading from BufferLoader.
* `ESS`: The effective sample size of the current sample set.

Instead of reading one at a time, the learning algorithm (booster) can read a small batch of examples from the BufferLoader at a time to take advantage of multi-threading in the subsequent computations.

* `LoadedData`: The array of the sampled training examples in memory. BufferLoader returns examples as reading from a circular queue, namely, the first example would be returned next after reading the last example.

#### Methods

Return next batch of examples

```
Procedure GetNextBatch():
  IF LoadingData IS READY DO
    LoadedData = LoadingData
    <Set LoadingData as NOT READY>
  END

  Batch = <Read next BatchSize examples from LoadedData>
  RETURN Batch
END
```

Update the scores of all examples after a new rule is added to the ensemble.

```
Procedure UpdateScores(Model):
  ModelSize = <Number of Trees in Model>
  FOR SampledExample IN LoadedData DO
    FOR Tree IN Model[SampledExample.LastTreeIndex...ModelSize] DO
      SampledExample.LastScore += Tree.Predict(SampledExample.LabeledData.Feature)
    END
    SampledExample.LastTreeIndex = ModelSize
  END
END
```

Update effective sample size of current sample.

```
Procedure UpdateESS():
  SumOfWeights = 0
  SumOfWeightSquared = 0
  FOR SampledExample IN LoadedData DO
    W = GetWeight(LoadedData.LabeledData.Label, LoadedData.LastScore)
    SumOfWeights += W
    SumOfWeightSquared += W * W
  END
  EffectiveSize = (SumOfWeights * SumOfWeights) / SumOfWeightSquared
  LoadedDataSize = <Number of entries in LoadedData>
  RETURN EffectiveSize / LoadedDataSize
END
```

A background thread keep receiving newly sampled examples from the stratified storage, and replace current sample once sufficient new samples are gathered.

```
Procedure StartSampledExamplesGatherer(SampledExamplesQueue):
  <Run this procedure in an independent thread>
  WHILE True DO
    WHILE <Size of LoadingData> < Size DO
      SampledExample = SampledExamplesQueue.BlockingRead()
      LoadingData.Append(SampledExample)
    END

    <Shuffle LoadingData>
    <Set LoadingData as READY>
  END
END
```


### StratifiedStorage

`StratifiedStorage` maintains a large volume of training examples
(more than the memory can fit) on disk.
It organizes examples into strata according to the weights of the examples with regard to the latest ensemble.

#### Structure

```
Stratum           := (IncomingMemBuffer, SlotIndices, OutgoingMemBuffer)
StratumMap        := Map(StratumIndex1 => Stratum1, StratumIndex2 => Stratum2, ...)
Strata            := (DiskBuffer, NumExamplesPerSlot, StratumMap)
StratifiedStorage := (Strata, WeightsTable, UpdatedExamplesQueue, SampledExamplesQueue, LastModelQueue)
```

At any time, a stratum maintains two in-memory buffer for each stratum, one for storing examples that are waiting to be written back to disk (`IncomingMemBuffer`), and the other for storing examples that are loaded into memory for the sampler (described below) to sample from (`OutgoingMemBuffer`).

* `DiskBuffer`: A file on disk that stores the majority of training examples.
* `NumExamplesPerSlot`: The capacity of a slot. The slots are basic unit for writing examples back to disk. The training examples are written into and reading out from the disk one block at a time.
* `WeightsTable`: The weight distribution of strata, constantly being updated.
* `UpdatedExamplesQueue`: The queue between the Samplers and the Selectors (described below) for sending training examples with updated scores.
* `SampledExamplesQueue`: The queue between the Samplers and the BufferLoader for sending sampled examples with regard to the latest model.
* `LastModelQueue`: The queue between the Boosting Learner and the Samplers to send the latest ensemble.


#### The Methods for Updating and Sampling in StratifiedStorage

```
Procedure StartStratifiedStorage():
  RunAssigner()
  RunSampler()
END

Procedure RunAssigner():
  <Run this procedure in an independent thread>
  WHILE True DO
    ScoredExample = UpdatedExamplesQueue.BlockingRead()
    Weight = GetWeight(ScoredExample.LabeledData.Label, ScoredExample.LastScore)
    Index = Log2(Weight)
    Strata.StratumMap[Index].IncomingMemBuffer.BlockingWrite(ScoredExample)
    WeightsTable[Index] += Weight
  END
END

Procedure RunSampler():
  <Run this procedure in an independent thread>
  LastGridVals = Map()
  WHILE True DO
    Index = <Sample a stratum using weighted sampling w.r.t. WeightsTable>
    OutgoingMemBuffer = Strata.StratumMap[Index].OutgoingMemBuffer
    LastGrid = LastGridVals[Index]
    GridSize = 2^(Index + 1)
    WHILE True DO
      ScoredExample = OutgoingMemBuffer.BlockingRead()
      OutdatedScore = ScoredExample.LastScore
      WeightsTable[Index] -= GetWeight(ScoredExample.LabeledData.Label, OutdatedScore)
      Model = LastModelQueue.Read()
      UpdatedScore = Model.Update(ScoredExample)
      UpdatedScoreExample = (ScoredExample.LabeledData, UpdatedScore, Size(Model))
      UpdatedExamplesQueue.Enqueue(UpdatedScoreExample)
      LastGrid += GetWeight(ScoredExample.LabeledData.Label, UpdatedScore)
      IF LastGrid >= GridSize DO
        SampledExample = UpdatedScoreExample
        BREAK
      END
    END
    WHILE LastGrid >= GridSize DO
      SampledExamplesQueue.Enqueue(SampledExample)
      LastGrid -= GridSize
    END
  END
END
```


#### Low Level Procedures for Maintaining a Single Stratum

There are two threads running for the strata, one for writing examples in the `IncomingMemBuffer` back to disk, the other for loading examples from disk to the `OutgoingMemBuffer`. Both queues have fixed size, so once they are full, the corresponding threads would pause and wait for their availabilities.

```
Procedure StratumEnqueue():
  <Run this procedure in an independent thread for every Stratum>
  Buffer = Array[]
  WHILE True DO
    Example = IncomingMemBuffer.BlockingRead()
    Buffer.Append(Example)
    IF LENGTH(Buffer) >= NumExamplesPerSlot DO
      <Write the examples in Buffer to a FREE block on Disk>
      <Mark the written block on disk as OCCUPIED>
      Buffer = Array[]
    END
  END
END


Procedure StratumDequeue():
  <Run this procedure in an independent thread for every Stratum>
  Buffer = Array[]
  WHILE True DO
    IF LENGTH(Buffer) == 0 DO
      Buffer = <Load examples from a OCCUPIED block from disk>
      <Mark the read block on disk as FREE>
    END
    Example = <Next example in Buffer>
    OutgoingMemBuffer.BlockingWrite(Example)
  END
END
```



### Booster

#### High Level Training Procedure

```
Procedure RunTraining(TotalIterations):
  NumIterations = 0
  Model = Array[]
  WHILE NumIterations < TotalIterations DO
    ExamplesBatch = BufferLoader.GetNextBatch()
    NewRule = UpdateWeakRules(ExamplesBatch)
    IF NewRule EXISTS DO
      Model.Append(NewRule)
      BufferLoader.UpdateScores(Model)
    END
    NumIterations += 1
  END
END
```

#### Details of the Weak Rules Updates

**Global Parameter**

* `gamma`: The target advantage of the valid weak rule

**Statistics Maintained by Each Weak Rule**

The null hypothesis imposed on each weak rule is that their advantage
with regard to current training data distribution is no larger than `gamma`.
Formally, we define a random varaibel $C_k$ for the weak rule $k$ as follows

$ C_k = \sum_i{i} Y_i \hat{Y}_i^{(k)} W_i - 2 \gamma W_i, $

where $Y_i$ and $W_i$ are the label and the weight of the example $i$,
and $\hat{Y}_i^{(k)}$ is the prediction given by the weak rule $k$ to the
example $i$.

To calculate the upper bound of $C_k$ using the stopping rule, we need to keep
track of following three variables for each weak rule

* `SumOfC`: The value of $C_k$
* `SumOfCSquared`: The sum of the squared of the upper bounds of the terms in the expression of $C_k$

To shrink `gamma` when no weak rule has required advantage, we also keep track of

* `SumOfScore`: The raw advantage of weak rules (see Pseudocode)

```
Procedure UpdateWeakRules(ExamplesBatch):
  IF <No weak rule detected after scanning over all examples in BufferLoader> DO
    Gamma = 0.9 * (MAX(SumOfScore) / BufferLoader.SumOfWeights / 2.0)
  END

  FOR EACH WeakRuleK DO
    FOR EACH Example IN ExamplesBatch DO
      Label = Example.LabeledData.Label
      Weight = GetWeight(Label, Example.LastScore)
      Prediction = WeakRuleK.Predict(Example.LabeledData.Feature)

      SumOfScore[WeakRuleK] += Prediction * Label * Weight
      SumOfC[WeakRuleK] += Prediction * Label * Weight - 2 * Gamma * Weight
      SumOfCSquared[WeakRuleK] += (Weight + 2.0 * Gamma * Weight)^2
    END
  END

  FOR EACH WeakRuleK DO
    Bound = GetBound(SumOfC[WeakRuleK], SumOfCSquared[WeakRuleK])
    IF SumOfC[WeakRuleK] > Bound DO
      RETURN WeakRuleK
    END
  END
  RETURN <DOES NOT EXIST A VALID RULE>
END

Procedure GetBound(SumOfC, SumOfCSquared):
  RETURN SQRT(3.0 * SumOfCSquared *
    (2.0 * LOG(LOG(3.0 * SumOfCSquared / ABS(SumOfC) / 2.0)) + Log(2.0 / Delta)))
END
```
