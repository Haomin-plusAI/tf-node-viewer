#Graph Viewer // gnv
#by Neil Tan

#https://stackoverflow.com/questions/28616512/can-a-python-script-access-variables-defined-in-an-interactive-session
#run -i view_node.py
#execfile('view_node.py')

# A Graph contains a set of tf.Operation objects, which represent units of computation; and tf.Tensor objects, which represent the units of data that flow between operations.
# https://www.tensorflow.org/api_docs/python/tf/Graph

import idx2numpy
import numpy as np
import tensorflow as tf
import os
from pathlib import Path

def init(graph, feed_dict=None):
    global __NVGRAPH__  # add this line!
    global __NVFEED__
    __NVGRAPH__ = graph
    __NVFEED__ = feed_dict

def ls(name = ''):
    graph = __NVGRAPH__
    if name != '':
        op = graph.get_operation_by_name(name)
        print(op)
        print("Output Tensor Names:")
        for output in op.outputs:
            print(output)

        print("")
        print("Output to: ")
        for it_node in graph.get_operations():
            for it_input in it_node.inputs:
                if str(op.name) in str(it_input):
                    print(it_node.name)
                    break

    else:
        for op in graph.get_operations():
            print(op.name)

def prepName(name):
    name = str(name).replace("/", "-")
    name = name.replace(":", "_")
    return name
        
def out(tName, path="./", outdir="out"):
    # print("=====Out Debug=====")
    # print("tName=" + tName)
    # print("path=" + path)
    # print("outdir=" + outdir)

    graph = __NVGRAPH__
    feed_dict = __NVFEED__

    #i.e. tName  = 'import/Variable_quint8_const:0'
    t = graph.get_tensor_by_name(tName)
    arr = t.eval(feed_dict)
    if arr.shape == (): #a work-around for idx2numpy, doesn't play well with single values
        arr = np.array([arr])
        # print(arr)
        # print(arr.shape)

    #string process tName: sub / and : for _
    #append .idx and use it for the file name

    outputName = prepName(tName) + ".idx"
    print("outputName: " + outputName)

    outPath = Path(path)
    if not outPath.exists():
        print("invalid path")
        return

    outPath = outPath / outdir

    if not outPath.exists():
        os.makedirs(outPath)
    elif outPath.exists() and not outPath.is_dir():
        print("invalid path")
        return

    outPath = outPath / outputName

    # print("outPath: " + str(outPath))
    f_write = open(str(outPath), 'wb')

    if t.dtype == tf.uint8 or t.dtype == tf.quint8:
        idx2numpy.convert_to_file(f_write, arr.astype(np.uint8))
    elif t.dtype == tf.int32 or t.dtype == tf.qint32:
        idx2numpy.convert_to_file(f_write, arr.astype(np.int32))
    else:
        idx2numpy.convert_to_file(f_write, arr.astype(np.float32))

    f_write.close()

def snap(opName, path="."):
    graph = __NVGRAPH__
    op = graph.get_operation_by_name(opName)

    outdir = prepName(opName)
    outPath = Path(path)
    if not outPath.exists():
        print("invalid path")
        return

    outPath = outPath / outdir
    if not outPath.exists():
        os.makedirs(outPath)
    elif outPath.exists() and not outPath.is_dir():
        print("invalid path")
        return
    
    print("========== in ==========")
    i = 0
    for it_input in op.inputs:
        print(str(i) + " : ", end='')
        out(it_input.name, str(outPath), "in")
        i += 1

    i = 0
    print("========== out =========")
    for it_output in op.outputs:
        print(str(i) + " : ", end='')
        out(it_output.name, str(outPath), "out")
        i += 1
