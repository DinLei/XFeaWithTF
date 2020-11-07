#!/usr/bin/env python
# coding=utf-8
"""
TensorFlow Implementation of <<DeepFM: A Factorization-Machine based Neural Network for CTR Prediction>> with the fellowing features：
#1 Input pipline using Dataset high level API, Support parallel and prefetch reading
#2 Train pipline using Coustom Estimator by rewriting model_fn
#3 Support distincted training using TF_CONFIG
#4 Support export_model for TensorFlow Serving

by lambdaji
"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import shutil
import tensorflow as tf
from datetime import date, timedelta

from utils.model_layer import emb_init
from utils.my_utils import set_dist_env, fea_identify
from utils.data_loader import input_fn_libsvm as input_fn

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("dist_mode", 0, "distribuion mode {0-loacal, 1-single_dist, 2-multi_dist}")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_threads", 16, "Number of threads")
tf.app.flags.DEFINE_integer("num_gpus", 1, "Number of threads")
tf.app.flags.DEFINE_integer("feature_size", 0, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 0, "Number of fields")
tf.app.flags.DEFINE_integer("multi_start_idx", -1, "multi_hot feature start position")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_integer("save_checkpoints_steps", 1000, "save checkpoint every step")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("train_dir", '', "training data dir")
tf.app.flags.DEFINE_string("test_dir", '', "test data dir")
tf.app.flags.DEFINE_string("valid_dir", '', "valid data dir")
tf.app.flags.DEFINE_string("xfea_conf", '', "feature conf file")
tf.app.flags.DEFINE_string("model_dir", '', "model check point dir")
tf.app.flags.DEFINE_string("dt_dir", '', "date partition dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")


def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    # ------hyperparameters----
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    # batch_norm_decay = params["batch_norm_decay"]
    # optimizer = params["optimizer"]
    layers = list(map(int, params["deep_layers"].split(',')))
    dropout = list(map(float, params["dropout"].split(',')))

    # ------bulid weights------
    FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
    # FM_W = tf.get_variable(name='fm_w', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    FM_W = emb_init(name='fm_w', feat_num=feature_size, initializer=tf.glorot_normal_initializer(),
                    zero_first_row=False)
    # FM_V = tf.get_variable(name='fm_v', shape=[feature_size, embedding_size],
    #                        initializer=tf.glorot_normal_initializer())
    FM_V = emb_init(name='fm_v', feat_num=feature_size, embedding_size=embedding_size,
                    initializer=tf.glorot_normal_initializer(), zero_first_row=False)

    # ------build feaure-------
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

    single_feat = feat_ids[:, 0:params.multi_start_idx]
    multi_feat = feat_ids[:, params.multi_start_idx:]

    # ------build f(x)------
    with tf.variable_scope("First-order"):
        feat_wgts = tf.nn.embedding_lookup(FM_W, feat_ids)  # None * F * 1
        y_w = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals), 1)

    with tf.variable_scope("Second-order"):
        embeddings = tf.nn.embedding_lookup(FM_V, feat_ids)  # None * F * K
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals)  # vij*xi
        sum_square = tf.square(tf.reduce_sum(embeddings, 1))
        square_sum = tf.reduce_sum(tf.square(embeddings), 1)
        y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)  # None * 1

    with tf.variable_scope("Deep-part"):
        if FLAGS.batch_norm:
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
            else:
                train_phase = False
        else:
            normalizer_fn = None
            normalizer_params = None

        deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])  # None * (F*K)
        for i in range(len(layers)):
            deep_inputs = tf.layers.dense(inputs=deep_inputs, units=layers[i],
                                          # kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                          activation=tf.nn.relu, name='mlp%d' % i)
            if FLAGS.batch_norm:
                # 放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
                deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' % i)
            if mode == tf.estimator.ModeKeys.TRAIN:
                # Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])
                # deep_inputs = tf.layers.dropout(inputs=deep_inputs, rate=dropout[i], training=mode ==
                # tf.estimator.ModeKeys.TRAIN)

        y_deep = tf.layers.dense(inputs=deep_inputs, units=1, activation=tf.identity,
                                 # kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                 name='deep_out')
        y_d = tf.reshape(y_deep, shape=[-1])

    with tf.variable_scope("DeepFM-out"):
        # y_bias = FM_B * tf.ones_like(labels, dtype=tf.float32)  # None * 1
        # warning;这里不能用label，否则调用predict/export函数会出错，train/evaluate正常；初步判断estimator做了优化，用不到label时不传
        y_bias = FM_B * tf.ones_like(y_d, dtype=tf.float32)  # None * 1
        y = y_bias + y_w + y_v + y_d
        pred = tf.sigmoid(y)

    predictions = {"prob": pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # ------bulid loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + \
           l2_reg * tf.nn.l2_loss(FM_W) + \
           l2_reg * tf.nn.l2_loss(FM_V)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    # ------bulid optimizer------

    if FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    else:
        # FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.layers.batch_normalization(x, momentum=FLAGS.batch_norm_decay, center=True, scale=True,
                                             updates_collections=None, is_training=True, reuse=None, name=scope_bn)
    bn_infer = tf.layers.batch_normalization(x, momentum=FLAGS.batch_norm_decay, center=True, scale=True,
                                             updates_collections=None, is_training=False, reuse=True, name=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


def build_estimator(multi_feats_range):
    """Build an estimator appropriate for the given model type."""
    set_dist_env(FLAGS)
    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.

    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "batch_norm_decay": FLAGS.batch_norm_decay,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout,
        "multi_start_idx": FLAGS.multi_start_idx,
        "multi_feats_range": multi_feats_range
    }

    session_config = tf.ConfigProto(device_count={'GPU': FLAGS.num_gpus, 'CPU': FLAGS.num_threads},
                                    inter_op_parallelism_threads=10,
                                    intra_op_parallelism_threads=10
                                    # log_device_placement=True
                                    )
    # 设置最小的GPU使用量
    session_config.gpu_options.allow_growth = True
    # 限制GPU资源使用：限制GPU使用率
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.32

    config = tf.estimator.RunConfig().replace(
        session_config=session_config,
        save_checkpoints_secs=FLAGS.save_checkpoints_steps,
        model_dir=FLAGS.model_dir,
        log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    model = tf.estimator.Estimator(model_fn=model_fn, params=model_params, config=config)

    return model


def main(_):
    # ------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')

    FLAGS.model_dir = os.path.join(FLAGS.model_dir, FLAGS.dt_dir)

    field_size, feature_size, _, multi_flag, _, _, multi_feats_range = fea_identify(FLAGS.xfea_conf)
    FLAGS.field_size = field_size
    FLAGS.feature_size = feature_size
    FLAGS.multi_start_idx = multi_flag

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('xfea_conf', FLAGS.xfea_conf)
    print('num_epochs ', FLAGS.num_epochs)
    print('feature_size ', FLAGS.feature_size)
    print('field_size ', FLAGS.field_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('dropout ', FLAGS.dropout)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('batch_norm_decay ', FLAGS.batch_norm_decay)
    print('batch_norm ', FLAGS.batch_norm)
    print('l2_reg ', FLAGS.l2_reg)

    # ------init Envs------
    tr_files = FLAGS.train_dir
    va_files = FLAGS.valid_dir
    te_files = FLAGS.test_dir

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    # ------build Tasks------
    DeepFM = build_estimator(multi_feats_range)
    task_type = set(FLAGS.task_type.split(","))

    if 'train' in task_type:
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
        test_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None,
            start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(DeepFM, train_spec, test_spec)
    if 'eval' in task_type:
        DeepFM.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))
    if 'infer' in task_type:
        preds = DeepFM.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size),
                               predict_keys="prob")
        with open(FLAGS.model_dir + "/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    if 'export' in task_type:
        feature_spec = {
            'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
            'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        DeepFM.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
