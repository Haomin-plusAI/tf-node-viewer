#!/usr/bin/env python3
# -*- coding:utf8 -*-
# pylint: disable=C0103
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
from view_node import GraphInspector, load_graph

mnist = read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)

graph = load_graph("./my-model/train.pb", "")
inspector = GraphInspector(graph, feed_dict={'x:0': mnist.test.images[0:10]})

inspector.ls()
inspector.ls("y_pred")
inspector.snap("y_pred")
