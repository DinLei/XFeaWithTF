#!/usr/bin/env python
# encoding: utf-8

# @author: ba_ding
# @contact: dinglei_1107@outlook.com
# @file: pb2pbtxt.py
# @time: 2020/9/16 11:24


import tensorflow as tf
from tensorflow.python.platform import gfile


# 函数功能能,将pb模型转换为pbtxt,转换好后存储到当前目录下,模型名字是protobuf.pbtxt
def convert_pb_to_pbtxt(pb_file, output_dir, output_name):
    with gfile.FastGFile(pb_file, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.io.write_graph(graph_def, output_dir, output_name+'.pbtxt', as_text=True)


# 函数功能能,将pb模型转换为txt,转换好后存储到txtmodel文件夹下,模型名字是frozen_model_test.txt
def pb_to_txt(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, "./txtmodel ", 'frozen_model_test.txt', as_text=True)


if __name__ == '__main__':
    # mygraph = pb_to_txt("frozen_darknet_yolov3_model.pb")
    pbpath = '../data/frozen_model.pb'
    convert_pb_to_pbtxt(pbpath, "../data", "t1")
