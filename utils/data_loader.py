#!/usr/bin/env python
# coding=utf-8

import os

import tensorflow as tf


def parse_function(example_proto, params):
    features = {
        'label': tf.FixedLenFeature([1], tf.float32),
        'label2': tf.FixedLenFeature([1], tf.float32),
        'cont_feats': tf.FixedLenFeature([params.cont_field_size], tf.float32),
        'vector_feats': tf.FixedLenFeature([params.vector_feats_size], tf.float32),
        'cate_feats': tf.FixedLenFeature(
            [params.cate_field_size + params.multi_feats_size + params.attention_feats_size], tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    labels = dict()
    label = parsed_features['label']
    label2 = parsed_features['label2']
    parsed_features.pop('label')
    parsed_features.pop('label2')
    labels['label'] = label
    labels['label2'] = label2
    return parsed_features, labels


def input_fn_tfrecord(file_dir_list, params):
    # data_set = tf.data.TFRecordDataset(file_dir_list, buffer_size=params.batch_size * params.batch_size) \
    #     .map(lambda x: parse_function(x, params), num_parallel_calls=1) \
    #     .shuffle(buffer_size=params.batch_size * 10) \
    #     .batch(params.batch_size, drop_remainder=True)

    files = tf.data.Dataset.list_files(file_dir_list, shuffle=False)
    data_set = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=10, block_length=1, sloppy=False)) \
        .map(lambda x: parse_function(x, params), num_parallel_calls=4) \
        .batch(params.batch_size) \
        .prefetch(4000)

    iterator = data_set.make_one_shot_iterator()
    feature_dict, labels = iterator.get_next()
    return feature_dict, labels


# 1 1:0.5 2:0.03519 3:1 4:0.02567 7:0.03708 8:0.01705 9:0.06296 10:0.18185 11:0.02497 12:1 14:0.02565 15:0.03267 17:0.0247 18:0.03158 20:1 22:1 23:0.13169 24:0.02933 27:0.18159 31:0.0177 34:0.02888 38:1 51:1 63:1 132:1 164:1 236:1
def input_fn_libsvm(data_dir, batch_size=32, shuffle_buffer_size=1024, num_epochs=1, perform_shuffle=False,
                    num_parallel_readers=4, hdfs=False):
    print('Parsing', data_dir)
    if hdfs:
        files_path = list_hdfs_dir(data_dir)
    else:
        files_path = list_local_dir(data_dir)

    def decode_libsvm(line):
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        id_vals = tf.reshape(splits.values, splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    filenames = tf.data.Dataset.list_files(files_path)

    dataset = filenames.apply(
        tf.data.experimental.parallel_interleave(tf.data.TextLineDataset,
                                                 cycle_length=num_parallel_readers, sloppy=True)
    )
    # dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(
    #     500000)  # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(decode_libsvm, num_parallel_calls=num_parallel_readers)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs).batch(batch_size).prefetch(batch_size)  # Batch size to use

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    # return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
    return batch_features, batch_labels


def list_local_dir(input_path):
    file_list = tf.gfile.ListDirectory(input_path)
    file_dir_list = []
    for file in file_list:
        if file[:4] == "part":
            file_path = os.path.join(input_path, file)
            file_dir_list.append(file_path)
    return file_dir_list


def list_hdfs_dir(path):
    files = []
    sample_dir = os.path.join(path, "part*")
    sample_dir_script = "hadoop fs -ls " + sample_dir + " | awk  -F ' '  '{print $8}'"
    for dir_path in os.popen(sample_dir_script).readlines():
        dir_path = dir_path.strip()
        files.append(dir_path)
    return files


if __name__ == '__main__':
    a = dict()
    a['liu'] = 'an'
    for key, value in a.items():
        print(key, " = ", value)
