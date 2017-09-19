python3 deep_mlp.py

/Users/neitan01/src/tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=./my-model/train.pb --input_checkpoint=./my-model/model.ckpt --output_graph=./graph_out/frozen_graph.pb  --output_node_names=y_pred

/Users/neitan01/src/tensorflow/bazel-bin/tensorflow/tools/quantization/quantize_graph --input=./graph_out/frozen_graph.pb --output_node_names="y_pred" --output=./graph_out/quantized_graph.pb --mode=eightbit

python3 export_tb.py

tensorboard --logdir=./log

python3 inference_8bit.py

ipython3
run inference_8bit.py
import view_node as nv
nv.init(tf.get_default_graph())
nv.ls()
nv.ls("import/y_pred")
nv.snap("import/y_pred")