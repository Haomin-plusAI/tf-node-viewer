# TensorFlow Node Viewer
## Introduction
  
  Node Viewer is an reverse engineering tool which provide the user an easy way to extract data from a graph. It is designed to be used in conjunction with Tensorboard. The Tensorboard presents the relation between ops and tensors, whereas, Node Viewer provide easy ways to export data to idx files.
  
  Here's what you can do with it:
  - inspect graphs
  - export all inputs/outputs of an op to idx files.
  - export tensors to idx files.
  
  
With `view_node.GraphInspector`, the following methods are available:
  - `ls()` Show All Ops in the graph.
    - `ls('node name')` Show All inputs and outputs of an Op/Node.
  - `export_tensor('tensor name')` export a tensor to a .idx file.
  - `snap('node name', outdir='./path')` export all input/output tensors to .idx files.

## Create and Train

With the mnist dataset, the deep_mlp.py creates a 3-layer fully-connected network with ReLU activation functions. The graph and trained parameters are written to `./my-model/train.pb`.

```
$ python3 deep_mlp.py
...
step 19500, training accuracy 0.94
step 19600, training accuracy 0.94
step 19700, training accuracy 1
step 19800, training accuracy 1
step 19900, training accuracy 1
test accuracy 0.9711

```
Regulation and Dropped-out are not employed in the interest of graph-simplicity

## Example of `node_view.GraphInspector`

The steps below can be performed in iPython or a script.

### Importing packages:

```
import tensorflow as tf
from view_node import GraphInspector, load_graph
```

### importing dataset:

```
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)
```

### importing graph:

```
graph = load_graph("./my-model/train.pb", name="")
```

### initialize GraphInspector:

```
inspector = GraphInspector(graph, feed_dict={"x:0": mnist.test.images[0:10]})
```
Or, you may use it without `feed_dict`:

```
inspector = GraphInspector(graph)
inspector.ls()
...
```
Note that some methods may not function properly without feed_dict.
Such as `snap`.

### Listing all the Ops in the graph:
```
inspector.ls()
"""
x
y
Layer1/Variable
Layer1/Variable/read
Layer1/Variable_1
...
Prediction/y_pred
Prediction/ArgMax/dimension
Prediction/ArgMax
...
"""
```

Inspecting an Op, the information displayed here is the same in Tensorboard:

```
inspector.ls("Prediction/y_pred")
"""
name: "Prediction/y_pred"
op: "ArgMax"
input: "OuputLayer/prediction"
input: "Prediction/y_pred/dimension"
attr {
  key: "T"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "Tidx"
  value {
    type: DT_INT32
  }
}

Output Tensor Names:
Tensor("Prediction/y_pred:0", shape=(?,), dtype=int64)

Output to:
Prediction/y_pred
"""
```

Here is a quickway to save all the input/output values of the Op to idx files:
```
inspector.snap("y_pred")
"""
Prediction/Equal
========== in ==========
0 : OuputLayer/prediction:0
outputName: OuputLayer-prediction_0.idx
1 : Prediction/y_pred/dimension:0
outputName: Prediction-y_pred-dimension_0.idx
========== out =========
0 : Prediction/y_pred:0
outputName: Prediction-y_pred_0.idx
"""
```

## Test

1. run `python3 deep_mlp.py`: generate testing `.pb` file
2. run `python3 test_list.py`: testing basic functionality

## (TODO) 8-bit Graph Quantization (Optional)
Use quantize_graph to convert `./my-model/train.pb` to an 8-bit quantized graph.

quantize_graph:
1. download the script `quantize_graph.py` from tensorflow github repo over [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/quantization/quantize_graph.py)
2. create a directory named `graph_logs`
3. run `python3 quantize_graph.py --input my-model/train.pb --mode=eightbit --output graph_out/quantized_graph.pb --output_node_names="Prediction/y_pred"`
```
$ ls graph_out
quantized_graph.pb
```
These steps will produce a quantized graph`./graph_out/quantized_graph.pb` for our example

### Create the Tensorboard Files for Quantized Graph

export_tb.py creates the log files using `./graph_out/quantized_graph.pb`

Tensorboard files are saved in `./log`
```
$ python3 export_tb.py
```

Run Tensorboard:
```
tensorboard --logdir=./log
```

### Using Node Viewer with the graph file

With ipython3, inference_8bit.py loads the graph in `./graph_out/quantized_graph.pb` which we will use with the node-viewer.

Setting the graph up with inference_8bit.py:

```
ipython3
run inference_8bit.py
Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
inference:    [7 2 1 0 4 1 4 9 6 9]
test labels:  [7 2 1 0 4 1 4 9 5 9]
```
Import node_viewer and supply the graph. Here, placeholder `x` is replaced with the first 10 entries of the test set.

## References

- [Stackoverflow: Access Variables in Session](https://stackoverflow.com/questions/28616512/can-a-python-script-access-variables-defined-in-an-interactive-session)
- [doc: Graph](https://www.tensorflow.org/api_docs/python/tf/Graph)