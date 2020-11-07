#!/usr/bin/env python
# encoding: utf-8

# @author: ba_ding
# @contact: dinglei_1107@outlook.com
# @file: test_graph.py
# @time: 2020/9/28 16:32

import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.platform import gfile


def create_graph():
    with tf.Session() as sess:
        a = tf.Variable(5.0, name='a')
        x = tf.Variable(6.0, name='x')
        b = tf.Variable(3.0, name='b')
        y = tf.add(tf.multiply(a, x), b, name="y")

        tf.global_variables_initializer().run()

        print(a.eval())  # 5.0
        print(x.eval())  # 6.0
        print(b.eval())  # 3.0
        print(y.eval())  # 33.0

        graph = convert_variables_to_constants(sess, sess.graph_def, ["y"])
        # writer = tf.summary.FileWriter("logs/", graph)
        tf.train.write_graph(graph, 'tms/', 'test_graph.pb', as_text=False)


def load_graph():
    with gfile.FastGFile("tms/test_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        output = tf.import_graph_def(graph_def, return_elements=['a:0', 'x:0', 'b:0', 'y:0'])
        # print(output)

    with tf.Session() as sess:
        result = sess.run(output)
        print(result)


if __name__ == "__main__":
    import sys

    flag = int(sys.argv[1])
    if flag == 1:
        create_graph()
    else:
        load_graph()
