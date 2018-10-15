# Sparrow Design and Pseudocode

## Table of Contents

* [System Design](#system-design)
* [Basic Data Types](#basic-data-types)
  * [Labeled Data](#labeleddata)
  * [Sampled Example](#sampledexample)
  * [Scored Example](#scoredexample)
  * [Tree](#tree)
* [System Entrance](#system-entrance)
* [StratifiedStorage](#stratifiedstorage)
* [Scanner](#scanner)
   * [BufferLoader](#bufferloader)
   * [Booster](#booster)

## System Design

![System Design](https://www.lucidchart.com/publicSegments/view/836caf8e-7af9-4af6-9a84-af70249ca153/image.png)

## Basic data types

There are two major data structure in Sparrow, one describes training/testing examples, the other describes an individual tree in the ensemble.

### LabeledData

#### Structure

An example consists of two components, Feature and Label.

```
LabeledData := (Feature, Label)
```

* **Feature**: Represents a set of binary features compiled for a raw example.
* **Label**: Either +1 or -1 (for binary classification)

In addition, there are two scenarios in which the examples can be associated with scores (predictions) from some
version of the trained model.


### ScoredExample

When an example is stored on disk (more strictly, in [StratifiedStorage](#stratifiedstorage)),
there is one score associated with it, which is the score predicted by lastest model when Sparrow accessed this
example last time.

Suppose when Sparrow accesses a specific example `x` for the first time, the lastest model `M` consists of `K` rules.
The example receives a score `M(x) = z`, sets `LastScore=z`, and `LastTreeIndex=K`.

Suppose when Sparrow accesses the example `x` for the second time, the latest model `M'` consists of `K'` rules.
Sparrow will update the score of `x` by querying only the recent `K'-LastTreeIndex` rules added to `M'` since `M`,
and thus speed-up the evaluation process.

```
ScoredExample := (LabeledData, LastScore, LastTreeIndex)
```


### SampledExample

The examples are represented in this format in BufferLoader (see [BufferLoader](#bufferloader)).
This is also the format used in the inner-loop of Sparrow, in which the [Booster](#booster) iterates over tables with SampledExample as rows.

There are two scores associated with the example. One is the score of the example when it was sampled, the other is its last updated score (the later is more recent than the former).

```
SampledExample := (LabeledData, SampledScore, SampledTreeIndex, LastScore, LastTreeIndex)
```

* **Score** is the weighted sum of the weak rule for this example
* **TreeIndex** is the index of the last weak rule added to a particular tree. The core is updated by adding to the old score the weighted sum of the rules added since then.
* There are two pairs of **Score,TreeIndex**:
   * `Sampled-` corresponds to time at which this example was sampled
   * `Last-` corresponds to the last time the weight of this example was updated.


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


## System Entrance

### Run Sparrow

There are three key components of Sparrow: the stratified storage, the buffered loader, and the learning algorithm.
They execute in separate threads in an asynchronous manner.

```
Procedure RunSparrow():
  StartStratifiedStorage()
  StartSampledExamplesGatherer()
  RunTraining()
END
```

All three components require a function for evaluating the weight of an example
given its label and its score predicted by the ensemble.

```
Procedure GetWeight(Label, Score):
  RETURN <the weight of the example, e.g., EXP(-Label * Score) for AdaBoost>
END
```


## StratifiedStorage

`StratifiedStorage` maintains a large volume of training examples
(more than the memory can fit) on disk.
It organizes examples into strata according to the weights of the examples with regard to the latest ensemble.

![](https://www.lucidchart.com/publicSegments/view/d832c410-9f5a-409c-8dc0-12597b9f17ad/image.png)

#### Structure

At showed in the diagram above, a stratum maintains two in-memory buffer for each stratum: one for storing examples that are waiting to be written back to disk (`InQueue`), and the other for storing examples that are loaded into memory for the samplers to sample from (`OutQueue`).

Strata for different weight ranges are organizes in a Map structure (`StratumMap`).

Lastly, there are three queue for message passing between different threads.

* `UpdatedExamplesQueue`: The queue for passing the training examples with updated scores between the Samplers
and the Assigners.
* `SampledExamplesQueue`: The queue for passing the sampled examples between the Samplers and the BufferLoader.
* `LastModelQueue`: The queue for passing the latest model between the Booster and the Samplers.

```
Stratum           := (InQueue, SlotIndices, OutQueue)
StratumMap        := Map(StratumIndex1 => Stratum1, StratumIndex2 => Stratum2, ...)
Strata            := (DiskBuffer, NumExamplesPerSlot, StratumMap)
StratifiedStorage := (Strata, WeightsTable, UpdatedExamplesQueue, SampledExamplesQueue, LastModelQueue)
```

* `DiskBuffer`: A file on disk that stores the majority of training examples.
* `NumExamplesPerSlot`: The capacity of a slot. The slots are basic unit for writing examples back to disk. The training examples are written into and reading out from the disk one block at a time.
* `WeightsTable`: The weight distribution of strata, constantly being updated.

The `WeightsTable` is a [lock-free map](https://github.com/jonhoo/rust-evmap),
i.e. even though it is shared among multiple threads, the read/write operations are both non-blocking.


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
    <Write ScoredExample to the InQueue of the corresponding stratum>
    <Update WeightsTable (non-blocking)>
  END
END

Procedure RunSampler():
  <Run this procedure in an independent thread>
  LastGridVals = Map()
  WHILE True DO
    Index = <Sample a stratum using weighted sampling w.r.t. WeightsTable>
    OutQueue = Strata.StratumMap[Index].OutQueue
    LastGrid = LastGridVals[Index]
    GridSize = 2^(Index + 1)
    WHILE True DO
      ScoredExample = OutQueue.BlockingRead()
      OutdatedScore = ScoredExample.LastScore
      Model = LastModelQueue.Read()
      UpdatedScore = Model.Update(ScoredExample)

      <Update WeightsTable (non-blocking)>
      <Send Updated Example to UpdatedExampleQueue>

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

There are two threads running for the strata, one for writing examples in the `InQueue` back to disk, the other for loading examples from disk to the `OutQueue`. Both queues have fixed size, so once they are full, the corresponding threads would pause and wait for their availabilities.

![](https://www.lucidchart.com/publicSegments/view/a4186e63-36be-4f2c-afee-65d973691d99/image.png)

```
Procedure StratumEnqueue():
  <Run this procedure in an independent thread for every Stratum>
  Buffer = Array[]
  WHILE True DO
    Example = InQueue.BlockingRead()
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
    OutQueue.BlockingWrite(Example)
  END
END
```

## Scanner

Scanner works in two threads.

1. The thread for the `BufferLoader` updates the examples in the BufferLoader from `StratifiedStorage`.
2. The thread for the `Booster` updates the statistics of the weak rules after reading in
the examples from the `BufferLoader`.


### BufferLoader

`BufferLoader` is an in-memory data loader to efficiently provide training examples to the learning algorithm. It maintains a weighted sample of the training dataset with regard to the latest ensemble model.
Internally, it constantly gather newly sampled examples from the stratified storage to replace its current sample set.

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
    <Query the new rules from LastTreeIndex to the latest>
    SampledExample.LastScore += <predictions from the new rules>
    SampledExample.LastTreeIndex = ModelSize
  END
END
```

Update effective sample size of current sample. In case that ESS is too small, Booster can stop training and waiting
for BufferLoader to gather a new sample set with a large ESS.

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

![](https://www.lucidchart.com/publicSegments/view/e864d382-6920-48bf-acb5-9389e15f7854/image.png)

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
