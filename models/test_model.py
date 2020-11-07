#!/usr/bin/env python
# encoding: utf-8

# @author: ba_ding
# @contact: dinglei_1107@outlook.com
# @file: test_model.py
# @time: 2020/10/14 14:03

if __name__ == "__main__":
    import tensorflow as tf

    a = tf.Variable("hello ", name="a")
    b = tf.Variable("tensorflow", name="b")
    result = tf.add(a, b, name="result")
    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        tf.io.write_graph(sess.graph_def, 'tms/', 'test_model.pb', as_text=False)
        print(result.eval())
