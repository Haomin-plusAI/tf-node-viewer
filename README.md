#TensorFlow Node Viewer
##Introduction
  
  Node Viewer is an reverse engineering tool which provide the user an easy way to extract data from a graph. It is designed to be used in conjunction with Tensorboard. The Tensorboard presents the relation between ops and tensors, whereas, Node Viewer provide easy ways to export data to idx files.
  
  Here's what you can do with it:
  
  - inspect graphs
  - export all inputs/outputs of an op to idx files.
  - export tensors to idx files.
  
  
The following functions are exposed:
  
  - `init(graph, feed_dict=None)` Specified the working graph and the feed_dict.
  - `ls()` Show All Ops in the graph.
  - `ls('node name')` Show All inputs and outputs of an Op/Node.
  - `out('tensor name')` export a tensor to a .idx file.
  - `snap('node name', outdir='./path')` export all input/output tensors to .idx files.

##Create and Train
With the mnist dataset, the deep_mlp.py creates a 3-layer fully-connected network with ReLU activation functions. The graph and trained parameters are written to `./my-model/train.pb`.

```
$ python3 deep_mlp.py

step 19500, training accuracy 0.94
step 19600, training accuracy 0.94
step 19700, training accuracy 1
step 19800, training accuracy 1
step 19900, training accuracy 1
test accuracy 0.9711

```
Regulation and Dropped-out are not employed in the interest of graph-simplicity

##8-bit Graph Quantization (Optional)
Use freeze_graph and quantize_graph to convert `./my-model/train.pb` to an 8-bit quantized graph.

`%tensorflow%` here refers to the root of the tensorflow repository.

freeze_graph:

```
$ mkdir graph_out
$ %tensorflow%/bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=./my-model/train.pb --input_checkpoint=./my-model/model.ckpt --output_graph=./graph_out/frozen_graph.pb  --output_node_names=y_pred
```
quantize_graph:

```
$ %tensorflow%/bazel-bin/tensorflow/tools/quantization/quantize_graph --input=./graph_out/frozen_graph.pb --output_node_names="y_pred" --output=./graph_out/quantized_graph.pb --mode=eightbit
$ ls graph_out
frozen_graph.pb    quantized_graph.pb
```
These step will produce a quantized graph`./graph_out/quantized_graph.pb` for our example

##Create the Tensorboard files

export_tb.py creates the log files using `./graph_out/quantized_graph.pb`
Tensorboard files are saved in `./log`

```
$ python3 export_tb.py
```
Run Tensorboard:

```
tensorboard --logdir=./log
```
##Using Node Viewer with the graph file

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

```
import view_node as nv
nv.init(tf.get_default_graph(), feed_dict={x: mnist.test.images[0:10]})
```

Listing all the Ops in the graph:

```
nv.ls()

...
import/MatMul_2_eightbit_requantize
import/MatMul_2
import/Variable_5
import/add_2
import/y_pred/dimension
import/y_pred
...
```
Inspecting an Op, the information displayed here is the same in Tensorboard:

```
nv.ls("import/y_pred")

name: "import/y_pred"
op: "ArgMax"
input: "import/add_2"
input: "import/y_pred/dimension"
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
Tensor("import/y_pred:0", shape=(?,), dtype=int64)

Output to:
import/y_pred
```

Here is a quickway to save all the input/output values of the Op to idx files:

```
nv.snap("import/y_pred")
```