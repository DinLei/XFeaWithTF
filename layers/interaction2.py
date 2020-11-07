#!/usr/bin/env python
# encoding: utf-8

# @author: ba_ding
# @contact: dinglei_1107@outlook.com
# @file: interaction2.py
# @time: 2020/9/14 17:07

import tensorflow as tf
from tensorflow.python.keras.layers import Layer


class AFMLayer(Layer):
    def __init__(self, attention_factor=4, l2_reg_w=0, dropout_rate=0, seed=1024, **kwargs):
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.dropout_rate = dropout_rate
        self.seed = seed
        super(AFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, training=None, **kwargs):
        pass

