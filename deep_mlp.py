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
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import time

FLAGS = None
keep_prob = 0.65

class MNIST_Generator(object):

  def __init__(self, data_dir):
    self._mnist = input_data.read_data_sets(data_dir, one_hot=True)

  def distort_image(self, input):
    image_2d = input.reshape((28,28))
    res = self.image_rand_rotate(image_2d, (-25, 25))
    res = self.image_rand_scale(res, scale=(0.5, 1.2))
    res = self.image_rand_transl(res, disp=(0.30, 0.30))
    res = self.image_rand_power(res, (1, 5))
    #res = image_rand_erode(res, kernel_size=(2,3))
    res = np.round(res)
    return image_2d.reshape(input.shape)

  def gen(self):
    num_images = self._images.shape[0]
    rand_index = np.random.random_integers(num_images - 1)
    label = self._labels[rand_index]
    image = self._images[rand_index]
    #mnist.train.images[np.random.random_integers(mnist.train.images.shape[0])]
    return (self.distort_image(image), label)

  def genTrainData(self):
    self._images = self._mnist.train.images
    self._labels = self._mnist.train.labels
    yield self.gen()

  def genTestData(self):
    self._images = self._mnist.test.images
    self._labels = self._mnist.test.labels
    yield self.gen()

  def image_frame(self, input, output_dim, fill=0):
    output = np.full(output_dim, fill, dtype=input.dtype)
    delta_x = (input.shape[0] - output_dim[0]) / 2
    delta_y = (input.shape[1] - output_dim[1]) / 2

    if(delta_x >= 0):
        src_x0 = delta_x
        src_x1 = src_x0 + output_dim[0]
        dst_x0 = 0
        dst_x1 = output_dim[0]
    else:
        src_x0 = 0
        src_x1 = input.shape[0]
        dst_x0 = np.absolute(delta_x)
        dst_x1 = output_dim[0] + delta_x
        
    src_x0 = int(np.floor(src_x0))
    src_x1 = int(np.floor(src_x1))
    dst_x0 = int(np.floor(dst_x0))
    dst_x1 = int(np.floor(dst_x1))
    
    if(delta_y >= 0):
        src_y0 = delta_y
        src_y1 = src_y0 + output_dim[1]
        dst_y0 = 0
        dst_y1 = output_dim[1]
    else:
        src_y0 = 0
        src_y1 = input.shape[1]
        dst_y0 = np.absolute(delta_x)
        dst_y1 = output_dim[1] + delta_x

    src_y0 = int(np.floor(src_y0))
    src_y1 = int(np.floor(src_y1))
    dst_y0 = int(np.floor(dst_y0))
    dst_y1 = int(np.floor(dst_y1))
    
    output[dst_x0:dst_x1, dst_y0:dst_y1] = input[src_x0:src_x1, src_y0:src_y1]
    
    return output

  def rotateImage(self, image, angle, fill=0):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
      #FIXME: warpAffine is a function of width, height, ordering matters
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return result

  def image_rand_rotate(self, input, angles=(-15, 15), fill=0):
    rand_angle = np.random.uniform(angles[0], angles[1])
    return self.rotateImage(input, rand_angle, fill)

  def image_rand_transl(self, input, disp=(0.25, 0.25), fill=0):
    h, w = input.shape
    rand_x = np.random.uniform(w * - disp[0], w * disp[0])
    rand_y = np.random.uniform(h * - disp[1], h * disp[1])
    M = np.float32([[1,0,rand_x],[0,1,rand_y]])
    return cv2.warpAffine(input,M,(w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)

  def image_rand_scale(self, input, scale=(0.5, 1.2), fill=0):
    h, w = input.shape
    rand_scale_factor = rand_x = np.random.uniform(scale[0], scale[1])
    output = cv2.resize(input,(int(np.ceil(rand_scale_factor * w)), int(np.ceil(rand_scale_factor * h))), interpolation = cv2.INTER_CUBIC)
    return self.image_frame(output, input.shape, fill)

  def image_rand_power(self, input, power=(1, 20)):
    rand_power = np.random.uniform(power[0], power[1])
    return input ** int(rand_power)

  def image_rand_erode(self, input, kernel_size=(3,5)):
    rand_size = np.round(np.random.uniform(kernel_size[0], kernel_size[1]))
    return cv2.erode(input, (rand_size, rand_size), iterations=1)


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
    ds_test = ds_test.batch(batch_size=FLAGS.batch_size)
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
      mon_sess.run(training_init_op)
      mon_sess.run(train_step)
      if global_step.eval(mon_sess) % 20 == 0:
        mon_sess.run(testing_init_op)
        print('test accuracy %g' % accuracy.eval(session=mon_sess))


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
