#!/usr/bin/env python
# encoding: utf-8

# @author: ba_ding
# @contact: dinglei_1107@outlook.com
# @file: io_utils.py
# @time: 2020/9/14 11:22


import tensorflow as tf
from multiprocessing import cpu_count


def decode_libsvm(value):
    columns = tf.string_split([value], ' ')

    label = tf.string_to_number(columns.values[0], out_type=tf.int32)
    label = tf.reshape(tf.cast(tf.equal(label, 1), tf.int32), [-1])

    splits = tf.string_split(columns.values[1:], ':')
    id_vals = tf.reshape(splits.values, splits.dense_shape)

    feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
    feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
    feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)

    return {'feat_ids': feat_ids, 'feat_vals': feat_vals}, label


def input_fn(data_file, num_epochs, shuffle_buffer_size, batch_size, num_parallel_readers):
    files = tf.data.Dataset.list_files(data_file)
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=num_parallel_readers, sloppy=False)
    )

    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(decode_libsvm, num_parallel_calls=cpu_count())

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    # 数据生产时间和数据消费时间相解耦
    dataset = dataset.prefetch(batch_size)

    dataset_iter = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, dataset_iter.initializer)
    return dataset_iter.get_next()
