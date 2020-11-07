# -*- coding: utf-8 -*-

import copy
import json
import os
import pickle
import time

import numpy as np
import pydoop.hdfs as hdfs
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '4'


def define_flags():
    flags = tf.app.flags
    tf.app.flags.DEFINE_string("task", "train", "train/dump/inference")
    tf.app.flags.DEFINE_string("data_dir", "hdfs://routerprd/user/predict/liulu/data/mlp/train/*",
                               "Set data path of train set. Coorpate with 'input-strategy DOWNLOAD'.")
    tf.app.flags.DEFINE_string("validate_dir", "hdfs://routerprd/user/predict/liulu/data/valid/dt=20200812/*",
                               "Set data path of validate set. Coorpate with 'input-strategy DOWNLOAD'.")
    tf.app.flags.DEFINE_string("train_dir", "./model", "Set model save path. Not input path")
    # tf.app.flags.DEFINE_string("log_dir", "./log", "Set tensorboard even log path.")
    # Flags Sina ML required: for tf.train.ClusterSpec
    tf.app.flags.DEFINE_string("ps_hosts", "",
                               "Comma-separated list of hostname:port pairs, you can also specify pattern like ps[1-5].example.com")
    tf.app.flags.DEFINE_string("worker_hosts", "",
                               "Comma-separated list of hostname:port pairs, you can also specify worker[1-5].example.co")
    # Flags Sina ML required:Flags for defining the tf.train.Server
    tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
    tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job.Sina ML required arg.")

    flags.DEFINE_string("checkpoints_dir", "/home/predict/net_disk_project/liulu/clk/mlp/checkpoints_dir",
                        "Set checkpoints path.")
    # flags.DEFINE_string("model_dir", "./model_dir", "Set checkpoints path.")
    flags.DEFINE_string("output_model", "./model_output", "Path to the training data.")
    flags.DEFINE_string("hidden_units", "128,64",
                        "Comma-separated list of number of units in each hidden layer of the NN")
    flags.DEFINE_integer("num_epochs", 1, "Number of (global) training epochs to perform, default 1")
    flags.DEFINE_integer("num_steps", 100000, "Number of (global) training steps to perform, default 1000000")
    flags.DEFINE_integer("batch_size", 512, "Training batch size, default 512")
    flags.DEFINE_integer("shuffle_buffer_size", 5000, "dataset shuffle buffer size, default 10000")
    flags.DEFINE_float("learning_rate", 0.01, "Learning rate, default 0.01")
    flags.DEFINE_float("dropout_rate", 0.25, "Drop out rate, default 0.25")
    flags.DEFINE_integer("num_parallel_readers", 40, "number of parallel readers for training data, default 5")
    flags.DEFINE_integer("save_checkpoints_steps", 5000, "Save checkpoints every this many steps, default 5000")
    flags.DEFINE_boolean("run_on_cluster", False,
                         "Whether the cluster info need to be passed in as input, default False")
    flags.DEFINE_integer("embedding_dim", 8, "100/200")

    FLAGS = flags.FLAGS
    return FLAGS


FLAGS = define_flags()


def parse_argument():
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")

    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)
    os.environ["TF_ROLE"] = FLAGS.job_name
    os.environ["TF_INDEX"] = str(FLAGS.task_index)

    # Construct the cluster and start the server
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = {"worker": worker_spec, "ps": ps_spec}
    os.environ["TF_CLUSTER_DEF"] = json.dumps(cluster)


def set_tfconfig_environ():
    if "TF_CLUSTER_DEF" in os.environ:
        cluster = json.loads(os.environ["TF_CLUSTER_DEF"])
        task_index = int(os.environ["TF_INDEX"])
        task_type = os.environ["TF_ROLE"]

        tf_config = dict()
        worker_num = len(cluster["worker"])
        if FLAGS.job_name == "ps":
            tf_config["task"] = {"index": task_index, "type": task_type}
            FLAGS.job_name = "ps"
            FLAGS.task_index = task_index
        else:
            if task_index == 0:
                tf_config["task"] = {"index": 0, "type": "chief"}
            else:
                tf_config["task"] = {"index": task_index - 1, "type": task_type}
            FLAGS.job_name = "worker"
            FLAGS.task_index = task_index

        if worker_num == 1:
            cluster["chief"] = cluster["worker"]
            del cluster["worker"]
        else:
            cluster["chief"] = [cluster["worker"][0]]
            del cluster["worker"][0]

        tf_config["cluster"] = cluster
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        print("TF_CONFIG", json.loads(os.environ["TF_CONFIG"]))


def my_model(features, labels, params):
    with tf.variable_scope('embedding_dict'):

        embedding_input = []
        feature_cols = tf.reshape(features['feat_ids'],
                                  [-1, len(_ONE_HOT_COLUMNS)])  # convert 3D to 2D

        for i, col_name in enumerate(_ONE_HOT_COLUMNS):
            init = tf.random_uniform([feature_size[col_name], params['embedding_dim']], -1.0, 1.0, dtype=tf.float32)
            # tf.random_normal(shape=[fea_size[col], params['embedding_dim']])

            weight = tf.get_variable('w_{}'.format(col_name), dtype=tf.float32, initializer=init)

            start_pos = 0
            if i > 0:
                start_pos = feature_max_index[_ONE_HOT_COLUMNS[i - 1]] + 1

            feature_col = tf.reshape(tf.slice(feature_cols, [0, i], [-1, 1]), [-1]) - start_pos
            embedding_input.append(tf.nn.embedding_lookup(weight, feature_col))

        embeddings = tf.concat(embedding_input, axis=1, name='embedding_concat')

    # MLP layers
    input_layers = [embeddings]
    for l_id, l_size in enumerate(params['hidden_units']):
        he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        layer = tf.layers.dense(input_layers[-1], l_size,
                                activation=tf.nn.relu,
                                name='hidden_layer_' + str(l_id),
                                kernel_regularizer=tf.contrib.layers.l1_regularizer(0.0),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(0.0),
                                kernel_initializer=he_init,
                                bias_initializer=tf.zeros_initializer())

        input_layers.append(layer)
        print('---------------------\nhidden_layer:\t' + str(l_id))

    y = tf.layers.dense(input_layers[-1], 1, name='y')
    prob = tf.nn.sigmoid(y)

    label = tf.reshape(labels, [-1, 1])
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=y, name='loss'))

    return prob, loss


def build_model(features, labels, mode, params):
    if num_gpus <= 1:
        prob, loss = my_model(features, labels, params)
    else:
        num_gpus_for_train = num_gpus - 1
        feature_splits = [{} for i in range(num_gpus_for_train)]
        _splits = {}
        label_splits = [{} for i in range(num_gpus_for_train)]
        for col_name in ['feat_ids', 'feat_vals']:
            _splits[col_name] = tf.split(features[col_name], num_gpus_for_train)
            for i in range(num_gpus_for_train):
                feature_splits[i][col_name] = _splits[col_name][i]

        label_splits = tf.split(labels, num_gpus_for_train)

        out_splits = []
        prob_splits = []

        for i in range(num_gpus_for_train):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i + 1)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                    prob_split, loss_split = my_model(feature_splits[i], label_splits[i], params)
                    prob_splits.append(prob_split)
                    out_splits.append(loss_split)

        prob = tf.reshape(tf.stack(prob_splits, axis=0), [-1, 1])
        loss = tf.reduce_mean(tf.stack(out_splits))

    pred = tf.reshape(prob, [-1, 1], name='prob')
    avg = tf.metrics.mean(prob)
    tf.summary.scalar('ctr_avg', avg[1])
    tf.summary.scalar('loss', loss)
    metrics = {'ctr': avg}
    hook_dict = {"loss": loss, 'ctr_avg': avg[1]}
    logging_hook = tf.train.LoggingTensorHook(hook_dict, every_n_iter=20)

    if mode == tf.estimator.ModeKeys.EVAL:
        label = tf.reshape(labels['label'], [-1, 1])
        auc = tf.metrics.auc(labels=label, predictions=pred)
        tf.summary.scalar('eval_loss', loss)
        tf.summary.scalar('pctr', avg)
        tf.summary.scalar('auc', auc)
        metrics = {'pctr': avg, 'auc': auc}
        hook_dict = {"eval_loss": loss, "pctr": avg[1], 'auc': auc[1]}
        logging_hook = tf.train.LoggingTensorHook(hook_dict, every_n_iter=20)
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, training_hooks=[logging_hook])

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    # AdagradOptimizer/AdamOptimizer
    train_op = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics,
                                      training_hooks=[logging_hook])


def build_estimator(model_dir):
    """Build an estimator appropriate for the given model type."""
    set_tfconfig_environ()
    hidden_units = [int(_) for _ in FLAGS.hidden_units.split(',')]
    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    session_config = tf.ConfigProto(device_count={'GPU': num_gpus},
                                    inter_op_parallelism_threads=0,
                                    intra_op_parallelism_threads=0)
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        session_config=session_config,
        save_checkpoints_secs=FLAGS.save_checkpoints_steps,  # 300
        keep_checkpoint_max=3,
        model_dir=model_dir)

    model = tf.estimator.Estimator(
        model_fn=build_model,
        params={
            'learning_rate': FLAGS.learning_rate,
            'dropout_rate': FLAGS.dropout_rate,
            'hidden_units': hidden_units,
            'embedding_dim': FLAGS.embedding_dim
        },
        config=run_config
    )

    return model


def one_hot_decode(list_column):
    return tf.string_to_number(tf.string_split([list_column], delimiter=':').values[0], out_type=tf.int32)


# def parse(value):
#     columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, field_delim=' ', use_quote_delim=False)
#     columns[0] = tf.string_to_number(columns[0], out_type=tf.int32)
#     for i in range(1, len(_COLUMNS)):
#         columns[i] = one_hot_decode(columns[i])
#     features = dict(zip(_COLUMNS, columns))
#     labels = features.pop('label')
#     # features['label'] = tf.to_float(labels)
#     return features, {'label': tf.to_float(labels)}


def parse_libsvm(value):
    columns = tf.string_split([value], ' ')

    label = tf.string_to_number(columns.values[0], out_type=tf.float32)
    label = tf.reshape(tf.cast(tf.equal(label, 1), tf.float32), [-1])

    splits = tf.string_split(columns.values[1:], ':')
    id_vals = tf.reshape(splits.values, splits.dense_shape)

    feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
    feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
    feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)

    return {'feat_ids': feat_ids, 'feat_vals': feat_vals}, label


def input_fn(data_file, num_epochs, shuffle_buffer_size, batch_size, num_parallel_readers):
    # """Generate an input function for the Estimator."""
    # assert tf.gfile.Exists(data_file), (
    #    '%s not found. Please make sure you have either run data_download.py or '
    #    'set both arguments --train_data and --test_data.' % data_file)

    files = tf.data.Dataset.list_files(data_file)
    # Extract lines from input files using the Dataset API.

    dataset = files.apply(
        tf.data.experimental.parallel_interleave(tf.data.TextLineDataset,
                                                 cycle_length=num_parallel_readers, sloppy=True)
    )
    # dataset = tf.data.TextLineDataset(files)

    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(parse_libsvm, num_parallel_calls=num_parallel_readers)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    # print("output_shapes: ", dataset.output_shapes)
    # print("output_types: ", dataset.output_types)
    # save the input tensor name of the graph
    # use for inference
    input_tensor_map = dict()
    dataset_iter = dataset.make_initializable_iterator()
    features, labels = dataset_iter.get_next()
    # print('features:' + str(features))
    for input_name, tensor in features.items():
        input_tensor_map[input_name] = tensor.name

    with open(os.path.join(FLAGS.checkpoints_dir, 'input_tensor_map.pickle'), 'wb') as f:
        pickle.dump(input_tensor_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, dataset_iter.initializer)

    return features, labels


def eval_input_fn(data_file, batch_size):
    # Extract lines from input files using the Dataset API.
    files = tf.data.Dataset.list_files(data_file)
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=1))
    # dataset = tf.data.TextLineDataset(files)
    dataset = dataset.map(parse_libsvm, num_parallel_calls=5)
    dataset = dataset.batch(batch_size)
    # print("output_shapes: ", dataset.output_shapes)
    # print("output_types: ", dataset.output_types)
    return dataset


def freeze_graph(output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if (output_node_names is None):
        output_node_names = 'loss'

    if not tf.gfile.Exists(my_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % my_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(my_dir)
    print(checkpoint)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        init_table = tf.tables_initializer(name="init_all_tables")
        sess.run(init_table)
        # for table_init_op in tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS):
        #    output_node_names += "," + table_init_op.name
        output_node_names += ",init_all_tables"

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")
        )  # # The output node names are used to select the usefull nodes

        # Finally we serialize and dump the output graph to the filesystem

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    return output_graph_def


def load_frozen_graph(prefix="frozen_graph"):
    frozen_graph_filename = os.path.join(my_dir, "frozen_model.pb")

    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:  # tf.Graph().as_default()
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        # tf.import_graph_def(graph_def, name=prefix)
        # tf.train.import_meta_graph(frozen_graph_filename)
        tf.import_graph_def(graph_def, name=prefix)

    return graph


def main(unused_argv):
    # Clean up the model directory if present
    # shutil.rmtree(FLAGS.checkpoints_dir, ignore_errors=True)

    model = build_estimator(FLAGS.checkpoints_dir)
    if isinstance(FLAGS.data_dir, str) and os.path.isdir(FLAGS.data_dir):
        train_files = [FLAGS.data_dir + '/' + x for x in os.listdir(FLAGS.data_dir)] if os.path.isdir(
            FLAGS.data_dir) else FLAGS.data_dir
    else:
        train_files = FLAGS.data_dir
    if isinstance(FLAGS.validate_dir, str) and os.path.isdir(FLAGS.validate_dir):
        eval_files = [FLAGS.validate_dir + '/' + x for x in os.listdir(FLAGS.validate_dir)] if os.path.isdir(
            FLAGS.validate_dir) else FLAGS.validate_dir
    else:
        eval_files = FLAGS.validate_dir

    print('train files: ' + str(train_files))
    print('eval files: ' + str(eval_files))

    # train process
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(train_files, FLAGS.num_epochs, FLAGS.shuffle_buffer_size,
                                  FLAGS.batch_size, FLAGS.num_parallel_readers),
        max_steps=FLAGS.num_steps
    )
    input_fn_for_eval = lambda: eval_input_fn(eval_files, FLAGS.batch_size)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=300)

    print("before train and evaluate")
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    print("after train and evaluate")
    # inference()
    # recall()
    # os.system("paste -d '\t' data/cos_sim.csv data/label.csv > data/prob_label.txt")
    # os.system("python evaluate.py")

    # Evaluate accuracy.
    results = model.evaluate(input_fn=input_fn_for_eval)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))
    print("after evaluate")
    '''
    if FLAGS.job_name == "worker" and FLAGS.task_index == 0:
        print("exporting model ...")
        feature_spec = tf.feature_column.make_parse_example_spec(_GLOBAL_FEATURES)
        print("feature spec: " + str(feature_spec))
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        model.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)
        # tf.contrib.predictor.from_saved_model(export_dir=)
    print("quit main")
    '''
    '''
    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        model.train(input_fn=lambda: input_fn(
            FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))
        results = model.evaluate(input_fn=lambda: input_fn(
            FLAGS.test_data, 1, False, FLAGS.batch_size))
        # Display evaluation metrics
        print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
        print('-' * 60)
        for key in sorted(results):
            print('%s: %s' % (key, results[key]))
    '''


def dump():
    # files = tf.data.Dataset.list_files('model2')
    # dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=1))

    with tf.Session() as sess:
        data_file = '/home/aps/net_disk/temp/checkpoints_dir/model.ckpt-0.meta'

        saver = tf.train.import_meta_graph(data_file, clear_devices=True)
        print('saver ok')
        saver.restore(sess, '/home/aps/net_disk/temp/checkpoints_dir/model.ckpt-0')

        graph = tf.get_default_graph()
        w = []
        bias = 0

        # list all trainable variables
        dim0 = 0
        # for var in tf.get_default_graph().get_collection(tf.GraphKeys.MODEL_VARIABLES):   #"trainable_variables"):
        print('\nglobal variables:')
        for var in tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            vv = sess.run(var)
            print(str(var.name) + '\t\t' + str(vv.shape))  # +str(vv))
            # print(vv)
            # idx = int(re.split('_|:', var.name)[1])
            # if var.name.split('/')[0] == 'weights':    #'weights/part_0:0':
            #    w.extend(list(vv[:,0]))
            #    dim0 += vv.shape[0]
            # if var.name == 'params/bias:0':
            #    bias = vv

        print('\nlocal variables:')
        for var in tf.get_default_graph().get_collection(tf.GraphKeys.LOCAL_VARIABLES):
            print(str(var.name) + '\t\t' + str(vv.shape))
        print('\ntrainable variables:')
        for var in tf.get_default_graph().get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            vv = sess.run(var)
            print(str(var.name) + '\t\t' + str(np.shape(vv)))
            # print(vv)
        print('\ndim: ' + str(dim0) + ', len(w):' + str(len(w)))


def evaluate():
    prefix = 'frozen_graph'
    # freeze_graph("prob")
    time0 = time.time()
    graph = load_frozen_graph(prefix=prefix)
    with tf.Session(graph=graph) as sess:
        sess.run(graph.get_operation_by_name(prefix + '/init_all_tables'))

        empty_X = {'feat_ids': [], 'feat_vals': []}
        label_list, hdfs_files = [], []

        for xfile in hdfs.ls(hdfs_dir):
            if hdfs.path.isdir(xfile):
                continue
            hdfs_files.append(xfile)

        pred_fp = open('pred', 'w')
        label_fp = open('label', 'w')
        print("Begin inference")
        for i in range(190, len(hdfs_files)):
            train_fp = hdfs.open(hdfs_files[i], 'rt')
            end_of_file = False
            while True:
                X_validate = copy.deepcopy(empty_X)
                read_line_num = 0

                while True:
                    line = train_fp.readline().strip().split(' ')
                    if len(line) != len(_COLUMNS):
                        end_of_file = True
                        break

                    X_validate['feat_ids'].append(list(map(lambda x: [int(x.split(':')[0])], line[1:])))
                    X_validate['feat_vals'].append(list(map(lambda x: [float(x.split(':')[1])], line[1:])))

                    label_list.append(line[0])

                    read_line_num += 1
                    if read_line_num == FLAGS.batch_size:
                        break

                input_feed = dict()
                input_feed[sess.graph.get_tensor_by_name(prefix + "/" + 'IteratorGetNext:0')] = X_validate['feat_ids']
                input_feed[sess.graph.get_tensor_by_name(prefix + "/" + 'IteratorGetNext:1')] = X_validate['feat_vals']

                prob = graph.get_operation_by_name(prefix + "/prob").outputs[-1]
                pred = sess.run(prob, feed_dict=input_feed)
                np.savetxt(pred_fp, pred, delimiter='\n', fmt='%s')

                if end_of_file:
                    break

        label_fp.writelines('\n'.join([str(x) for x in label_list]) + '\n')
        train_fp.close()
        pred_fp.close()
        label_fp.close()
        os.system("paste -d '\t' pred label  > prob_label")
        os.system("python evaluate.py prob_label")
        time1 = time.time()
        print("evaluate cost: ", time1 - time0)


def fea_identify(xfea_conf_dir):
    """
    Read the xfea features_list conf and identify the xfea index range of features
    Return: field_size, feature_size, feature_values, feat_name-map-position_range
    """
    with open(xfea_conf_dir, 'r') as xfea_conf:
        fea_size = {}
        fea_dict = {}
        fea_slot = {}
        start_pos = 0
        for line in xfea_conf:
            line = line.strip()
            if not line:
                continue
            if line[0] == '#' or line[0] == '//':
                continue
            line = line.strip().split(';')
            value_size = 1
            fea_name = ''
            for l in line:
                if 'name' in l:
                    fea_name = l.split('=')[1]
                if 'slot' in l:
                    slot = int(l.split('=')[1])
                if 'feat_values' in l:
                    value_size = len(l.split(','))
                if 'hash_range_max' in l:
                    value_size = int(l.split('=')[1])

            if fea_name != '':
                fea_slot[fea_name] = slot
                fea_size[fea_name] = value_size
                fea_dict[fea_name] = (start_pos, start_pos + value_size - 1)
                start_pos += value_size

    cols = [_[0] for _ in sorted(fea_slot.items(), key=lambda item: item[1])]

    return len(cols), start_pos + 1, cols, fea_size, fea_dict


if __name__ == '__main__':
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    if FLAGS.run_on_cluster: parse_argument()
    tf.logging.set_verbosity(tf.logging.INFO)

    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    my_dir = FLAGS.checkpoints_dir
    hdfs_dir = 'hdfs://routerprd/user/predict/liulu/data/valid/dt=20200812'
    xfea_conf_dir = '/home/predict/net_disk_project/liulu/clk/gds_cvr/conf/suning_features_list_v3.conf'
    feature_max_index, feature_size, _ONE_HOT_COLUMNS = fea_identify(xfea_conf_dir)

    #     for key in feature_max_index.keys():
    #         print("%s: %d, %d" % (key, feature_max_index[key], feature_size[key]))

    _COLUMNS = ['label'] + _ONE_HOT_COLUMNS

    if FLAGS.task == 'dump':
        dump()
    elif FLAGS.task == 'evaluate':
        evaluate()
    else:
        tf.app.run(main=main)
