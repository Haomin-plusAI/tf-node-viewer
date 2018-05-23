import cv2
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

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

  def next_batch(self, batch_size):
    batch_data = np.zeros([batch_size, self._images.shape[1]])  #TODO: extendable dimensions
    batch_label = np.zeros([batch_size, self._labels.shape[1]])

    for i in range(batch_size):
      data, label = self.gen()
      batch_data[i, :] = data
      batch_label[i, :] = label
    
    return (batch_data, batch_label)

  def next_train_batch(self, batch_size):
    self._images = self._mnist.train.images
    self._labels = self._mnist.train.labels
    return self.next_batch(batch_size)

  def next_test_batch(self):
    self._images = self._mnist.test.images
    self._labels = self._mnist.test.labels
    return self.next_batch(self._mnist.test.images.shape[0])

  def getNumTrain(self):
    return self._mnist.train.images.shape[0]

  def getNumTest(self):
    return self._mnist.test.images.shape[0]

  def genTestData(self):
    self._images = self._mnist.test.images
    self._labels = self._mnist.test.labels
    yield self.gen()

  def image_frame(self, input, output_dim, fill=0):
    output = np.full(output_dim, fill, dtype=input.dtype)
    delta_x = (input.shape[0] - output_dim[0]) / 2.0 #true div, python2 compatibility
    delta_y = (input.shape[1] - output_dim[1]) / 2.0

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
