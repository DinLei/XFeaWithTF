# -*- coding:utf-8 -*-
"""

Authors:
    Weichen Shen,wcshen1994@163.com,
    Harshit Pande

"""

import itertools

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import (Zeros, glorot_normal)
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2

from common.layer_utils import reduce_sum, softmax


class AFMLayer(Layer):
    """
    Attentonal Factorization Machine models pairwise (order-2) feature
    interactions without linear term and bias.

      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      Arguments
        - **attention_factor** : Positive integer, dimensionality of the
         attention network output space.

        - **l2_reg_w** : float between 0 and 1. L2 regularizer strength
         applied to attention network.

        - **dropout_rate** : float between in [0,1). Fraction of the attention net output units to dropout.

        - **seed** : A Python integer to use as random seed.

      References
        - [Attentional Factorization Machines : Learning the Weight of Feature
        Interactions via Attention Networks](https://arxiv.org/pdf/1708.04617.pdf)
    """

    def __init__(self, attention_factor=4, l2_reg_w=0, dropout_rate=0, seed=1024, **kwargs):
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.dropout_rate = dropout_rate
        self.seed = seed
        super(AFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 这里 input_shape 是第一次运行 call() 时参数 inputs 的形状
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))

        embedding_size = int(input_shape[-1])

        self.attention_W = self.add_weight(
            shape=(embedding_size, self.attention_factor),
            initializer=glorot_normal(seed=self.seed),
            regularizer=l2(self.l2_reg_w),
            name="attention_W")
        self.attention_b = self.add_weight(
            shape=(self.attention_factor,),
            initializer=Zeros(),
            name="attention_b")
        self.projection_h = self.add_weight(
            shape=(self.attention_factor, 1),
            initializer=glorot_normal(seed=self.seed),
            name="projection_h")
        self.projection_p = self.add_weight(
            shape=(embedding_size, 1),
            initializer=glorot_normal(seed=self.seed),
            name="projection_p")

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed)

        self.tensordot = tf.keras.layers.Lambda(lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

        # Be sure to call this somewhere!
        super(AFMLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        # inputs的shape：[batch_size, ]
        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embeds_vec_list = inputs
        row = []
        col = []

        # 交叉对 数量： m = num * (num - 1) / 2
        for r, c in itertools.combinations(embeds_vec_list, 2):
            row.append(r)
            col.append(c)

        # 把所有 m 个 embed向量 (1 * embed_size) 接起来：(1 * (m * embed_size))
        p = tf.concat(row, axis=1)
        q = tf.concat(col, axis=1)
        inner_product = p * q

        # 为什m这里不进行reshape：tf.reshape(inner_product, shape=[-1, embed_size])
        # ？？？可能输入理解有误，继续阅读
        bi_interaction = inner_product
        attention_temp = tf.nn.relu(
            tf.nn.bias_add(
                # 这里的tensordot就是 左边矩阵的最后一维 与 右边矩阵的第一维 做矩阵相乘
                # attention_W 的 维度 是 (embed_size * attention_factor)
                tf.tensordot(bi_interaction, self.attention_W, axes=(-1, 0)),
                self.attention_b
            )
        )
        #  Dense(self.attention_factor,'relu',kernel_regularizer=l2(self.l2_reg_w))(bi_interaction)
        self.normalized_att_score = softmax(
            # attention_temp 的维度：(交叉数量m, attention_factor)
            # projection_h 的维度：(attention_factor, 1)
            tf.tensordot(attention_temp, self.projection_h, axes=(-1, 0)),
            dim=1
        )
        # axis = 1, 最后累加结果保持最外面的维度，这里应该是batch操作？？？
        attention_output = reduce_sum(
            self.normalized_att_score * bi_interaction, axis=1)

        attention_output = self.dropout(attention_output, training=training)  # training

        afm_out = self.tensordot([attention_output, self.projection_p])
        return afm_out

    def compute_output_shape(self, input_shape):

        if not isinstance(input_shape, list):
            raise ValueError('A `AFMLayer` layer should be called '
                             'on a list of inputs.')
        return (None, 1)

    def get_config(self, ):
        config = {'attention_factor': self.attention_factor,
                  'l2_reg_w': self.l2_reg_w, 'dropout_rate': self.dropout_rate, 'seed': self.seed}
        base_config = super(AFMLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):

        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum = tf.square(reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)
