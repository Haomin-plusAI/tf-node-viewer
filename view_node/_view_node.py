# -*- coding:utf8 -*-
# Graph Viewer by Neil Tan, Dboy Liao
import os
import sys
import idx2numpy
import numpy as np
import tensorflow as tf
from pathlib import Path

__all__ = ["GraphInspector"]


class GraphInspector(object):

    def __init__(self, graph, feed_dict=None):
        """
        Initialize Inspection Context

        Arguments
        =========
        - graph <`tf.Graph`>: the graph to inspect
        - feed_dict <`dict`>: the default feed dictionary for the graph
        """
        assert isinstance(graph, tf.Graph), \
            "Expecting {}, get {}".format(tf.Graph, type(graph))
        self._graph = graph
        self._feed_dict = feed_dict

    def _prepare_name(self, name):
        name = str(name).replace("/", "-")
        name = name.replace(":", "_")
        return name

    def ls(self, op_name=''):
        """
        Show Graph

        Arguments
        =========
        - op_name <`str`>: the root operation name to list. List all operations
            in the graph if op_name is ''
        """
        if not op_name == '':
            op = self._graph.get_operation_by_name(op_name)
            print(op)
            print("Output Tensor Names:")
            for output in op.outputs:
                print(output)
            print("")
            print("Output to: ")
            for it_node in self._graph.get_operations():
                for it_input in it_node.inputs:
                    if str(op.name) in str(it_input):
                        print(it_node.name)
                        break
        else:
            for op in self._graph.get_operations():
                print(op.name)

    def export_tensor(self, node_name, path="./", outdir="out"):
        """
        Export Tensor in Graph

        Arguments
        =========
        - node_name <`str`>: the name of node to export
        - path <`str`>: the root path, default './'
        - outdir <`str`>: the output directory name under root path, default 'out'

        Returns
        =======
        `bool`: `True` if success, `False` otherwise
        """
        print(node_name)
        # i.e. tName  = 'import/Variable_quint8_const:0'
        t = self._graph.get_tensor_by_name(node_name)
        with tf.Session(graph=self._graph) as sess:
            tf.global_variables_initializer().run()
            arr = t.eval(self._feed_dict)

        # a work-around for idx2numpy, doesn't play well with single values
        if arr.shape == ():
            arr = np.array([arr])

        # string process tName: sub / and : for _
        # append .idx and use it for the file name
        outputName = self._prepare_name(node_name) + ".idx"
        print("outputName: " + outputName)

        outPath = Path(path)
        if not outPath.exists():
            print("invalid path")
            return False

        outPath = outPath / outdir

        if not outPath.exists():
            os.makedirs(outPath)
        elif outPath.exists() and not outPath.is_dir():
            print("invalid path")
            return False

        outPath = outPath / outputName

        # print("outPath: " + str(outPath))
        with open(str(outPath), 'wb') as fid:
            if t.dtype == tf.uint8 or t.dtype == tf.quint8:
                idx2numpy.convert_to_file(fid, arr.astype(np.uint8))
            elif t.dtype == tf.int32 or t.dtype == tf.qint32:
                idx2numpy.convert_to_file(fid, arr.astype(np.int32))
            else:
                idx2numpy.convert_to_file(fid, arr.astype(np.float32))
        return True

    def snap(self, op_name, path="."):
        """
        Snapshot Operation

        Arguments
        =========
        - op_name <`str`>: the name of target operation
        - path <`str`>: the root path of output files

        Returns
        =======
        `bool`: `True` if success, `False` otherwise
        """
        op = self._graph.get_operation_by_name(op_name)

        outdir = self._prepare_name(op_name)
        outPath = Path(path)
        if not outPath.exists():
            print("invalid path")
            return False

        outPath = outPath / outdir
        if not outPath.exists():
            os.makedirs(outPath)
        elif outPath.exists() and not outPath.is_dir():
            print("invalid path")
            return False
    
        print("========== in ==========")
        for i, it_input in enumerate(op.inputs):
            sys.stdout.write(str(i) + " : ")
            if not self.export_tensor(it_input.name, str(outPath), "inputs"):
                return False

        print("========== out =========")
        for i, it_output in enumerate(op.outputs):
            sys.stdout.write(str(i) + " : ")
            if not self.export_tensor(it_output.name, str(outPath), "outputs"):
                return False
