#Neil Tan
#A script that loads a saved graph and produce Tensorboard files bases on it

import tensorflow as tf
from tensorflow.python.platform import gfile


with tf.Session() as sess:
    with gfile.FastGFile('./graph_out/quantized_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
        graph = tf.get_default_graph()


        #Get a list of operations
        for op in graph.get_operations():
            #Prints the tensor that a given operation produces
            print(op.values())

            #tensorboard graph visualization        
            LOGDIR='./log'
            train_writer = tf.summary.FileWriter(LOGDIR)
            train_writer.add_graph(sess.graph)