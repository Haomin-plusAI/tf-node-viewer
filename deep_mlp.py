# This script is based on:
# https://www.tensorflow.org/get_started/mnist/pros
#
# References:
# https://www.tensorflow.org/versions/master/api_guides/python/input_dataset
# https://www.tensorflow.org/versions/master/performance/datasets_performance
# https://www.tensorflow.org/versions/master/programmers_guide/datasets
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
# https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428

import argparse
import sys
import tensorflow as tf
from sample_generator import MNIST_Generator

FLAGS = None
keep_prob = 0.65


def deepnn(x):
  with tf.name_scope("Layer1"):
    W_fc1 = weight_variable([784, 128], name='W_fc1')
    b_fc1 = bias_variable([128], name='b_fc1')
    a_fc1 = tf.add(tf.matmul(x, W_fc1), b_fc1, name="zscore")
    h_fc1 = tf.nn.relu(a_fc1)
    layer1 = tf.nn.dropout(h_fc1, keep_prob)

  with tf.name_scope("Layer2"):
    W_fc2 = weight_variable([128, 64], name='W_fc2')
    b_fc2 = bias_variable([64], name='b_fc2')
    a_fc2 = tf.add(tf.matmul(layer1, W_fc2), b_fc2, name="zscore")
    h_fc2 = tf.nn.relu(a_fc2)
    layer2 = tf.nn.dropout(h_fc2, keep_prob)
  
  with tf.name_scope("OuputLayer"):
    W_fc3 = weight_variable([64, 10], name='W_fc3')
    b_fc3 = bias_variable([10], name='b_fc3')
    y_pred = tf.add(tf.matmul(layer2, W_fc3), b_fc3, name="prediction")

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

  global_step = tf.train.get_or_create_global_step(graph=tf.get_default_graph())
  
  with tf.name_scope("Pipeline"):
    # Import data
    mnist_inputPipe = MNIST_Generator(FLAGS.data_dir)
    ds_train = tf.data.Dataset.from_generator(
      mnist_inputPipe.genTrainData, (tf.float32, tf.float32), (tf.TensorShape([784]), tf.TensorShape([10])))
    #ds = ds.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    ds_train = ds_train.repeat()
    ds_train = ds_train.batch(batch_size=FLAGS.batch_size)
    ds_test = tf.data.Dataset.from_generator(
      mnist_inputPipe.genTestData, (tf.float32, tf.float32), (tf.TensorShape([784]), tf.TensorShape([10])))
    ds_test = ds_test.repeat()
    ds_test = ds_test.batch(mnist_inputPipe.getNumTest())
    iterator = tf.data.Iterator.from_structure(ds_train.output_types, ds_train.output_shapes)
    training_init_op = iterator.make_initializer(ds_train)
    testing_init_op = iterator.make_initializer(ds_test)
    
  (x, y_) = iterator.get_next()

  tf.summary.image(
    "input images",
    tf.reshape(x, [-1, 28, 28, 1]),
    max_outputs=8)

  # Build the graph for the deep net
  y_pred = deepnn(x)

  with tf.name_scope("Loss"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, 
                                                            logits=y_pred)
    loss = tf.reduce_mean(cross_entropy, name="cross_entropy_loss")
  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, tf.train.get_global_step(), name="train_step")
  
  with tf.name_scope("Prediction"): 
    correct_prediction = tf.equal(tf.argmax(y_pred, 1, name='y_pred'), 
                                  tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
    tf.summary.scalar('accuracy', accuracy)

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.chk_dir,
      hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
             tf.train.NanTensorHook(loss),
             tf.train.SessionRunHook()],
      config=tf.ConfigProto(
          log_device_placement=FLAGS.log_device_placement)) as mon_sess:

    while not mon_sess.should_stop():
      mon_sess.run([training_init_op, train_step])
      #mon_sess.run(train_step)
      current_step = global_step.eval(session=mon_sess)
      if current_step % 10 == 0:
        mon_sess.run(testing_init_op)
        print('step %d, test accuracy %g' % (current_step, accuracy.eval(session=mon_sess)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--chk_dir', type=str,
                      default='./checkpoint',
                      help='Directory for storing check point')
  parser.add_argument('--batch_size', type=int,
                      default=100,
                      help='batch size, default = 100')
  parser.add_argument('--max_steps', type=int,
                      default=1000000,
                      help='Number of batches to run')
  parser.add_argument('--log_device_placement', type=bool,
                      default=False,
                      help='Whether to log device placement.')
  parser.add_argument('--log_frequency', type=int,
                      default=10,
                      help='How often to log results to the console.')


  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
