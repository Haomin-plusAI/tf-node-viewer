#Neil Tan
#This script loads the quantized graph and performs inference bases on it

import tensorflow as tf
import argparse
from tensorflow.python.platform import gfile
from tensorflow.examples.tutorials.mnist import input_data


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
FLAGS, unparsed = parser.parse_known_args()

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

sess = tf.InteractiveSession()

with gfile.FastGFile('./graph_out/quantized_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    g_in = tf.import_graph_def(graph_def)
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name('import/Placeholder:0')
    y = graph.get_tensor_by_name('import/y_pred:0')
    
    #y_: mnist.test.labels
    print("inference:   ", y.eval(feed_dict={
        x: mnist.test.images[0:10]}))

    print("test labels: ", tf.argmax(mnist.test.labels[0:10], 1).eval())

