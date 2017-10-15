# -*- coding:utf8 -*-
import os
import tensorflow as tf
from google.protobuf import text_format

__all__ = ["load_graph"]


def load_graph(path, name=None):
    """
    Load Graph by Given Protobuf File

    Arguments
    =========
    - path <`str`>: path of the protobuf file
    - name <`str`>: name of the imported graph

    Returns
    =======
    - graph <`tf.Graph`>: a tensorflow Graph
    """
    _, ext = os.path.splitext(path)
    if ext == ".pb":
        graph_def = _parse_pb_file(path)
    elif ext == ".pbtxt":
        graph_def = _parse_pbtxt_file(path)
    else:
        raise ValueError("Unknown file extension: {}".format(ext))
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name=name)
    return graph


def _parse_pb_file(pb_path):
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_path, "rb") as fid:
        graph_def.ParseFromString(fid.read())
    return graph_def


def _parse_pbtxt_file(pbtxt_path):
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(pbtxt_path, "r") as fid:
        text_format.Parse(fid.read(), graph_def)
    return graph_def
