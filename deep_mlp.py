#This script is based on:
#https://www.tensorflow.org/get_started/mnist/pros

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.tools import freeze_graph

import tensorflow as tf

FLAGS = None


def deepnn(x):


    #https://mxnet.incubator.apache.org/tutorials/python/mnist.html

    W_fc1 = weight_variable([784, 128], name='W_fc1')
    b_fc1 = bias_variable([128], name='b_fc1')
    a_fc1 = tf.matmul(x, W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(a_fc1)

    W_fc2 = weight_variable([128, 64], name='W_fc2')
    b_fc2 = bias_variable([64], name='b_fc2')
    a_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
    h_fc2 = tf.nn.relu(a_fc2)

    W_fc3 = weight_variable([64, 10], name='W_fc3')
    b_fc3 = bias_variable([10], name='b_fc3')
    y_pred = tf.matmul(h_fc2, W_fc3) + b_fc3

    return y_pred


def weight_variable(shape, name):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name)


def bias_variable(shape, name):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name)


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_pred = deepnn(x)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pred))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_pred, 1, name='y_pred'), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    graph_path = tf.train.write_graph(sess.graph_def, './my-model', 'train.pb')
    print('written graph to: %s' % graph_path)

    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1]})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels}))
    
    saver.save(sess, "./my-model/model.ckpt")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

#/Users/neitan01/src/tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=./my-model/train.pb --input_checkpoint=./my-model/model.ckpt --output_graph=./graph_out/frozen_graph.pb  --output_node_names=y_pred
#/Users/neitan01/src/tensorflow/bazel-bin/tensorflow/tools/quantization/quantize_graph --input=./graph_out/frozen_graph.pb --output_node_names="y_pred" --output=./graph_out/quantized_graph.pb --mode=eightbit