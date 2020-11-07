#!/usr/bin/env python
# coding=utf-8

import datetime
import json
import os


def arg_parse(argv):
    parse_dict = dict()
    for i in range(1, len(argv)):
        line_parse = argv[i].split("=")
        key = line_parse[0].strip()
        value = line_parse[1].strip()
        parse_dict[key] = value
    return parse_dict


def shift_date_time(dt_time, offset_day, time_structure='%Y%m%d'):
    dt = datetime.datetime(int(dt_time[0:4]), int(dt_time[4:6]),
                           int(dt_time[6:8]))
    delta = datetime.timedelta(days=offset_day)
    del_day_date = dt + delta
    del_day_time = del_day_date.strftime(time_structure)
    return del_day_time


def shift_hour_time(dt_time, offset_hour, time_structure='%Y%m%d%H'):
    dt = datetime.datetime(int(dt_time[0:4]), int(dt_time[4:6]),
                           int(dt_time[6:8]), int(dt_time[8:10]))
    delta = datetime.timedelta(hours=offset_hour)
    del_date = dt + delta
    del_time = del_date.strftime(time_structure)
    return del_time


def feat_size(path, alg_name):
    cont_size = 0
    vector_size = 0
    cate_size = 0
    multi_feats_size = 0
    multi_cate_field = 0
    attention_feats_size = 0
    multi_cate_range = []
    cate_range = []
    attention_range = []
    attention_cate_field = []
    vec_name_list = ["user_vec", "ruUserVec", "item_vec", "user_kgv", "item_kgv"]
    no_pool_alg = ["dnn", "deepfm"]
    attention_alg = ["din", "dinfm", "dien"]

    files = os.listdir(path)
    for file in files:
        if file != "dnn.conf" and file != "lr.conf":
            continue

        file_path = path + "/" + file
        print("----read %s----" % file_path)
        with open(file_path, 'r') as f:
            index_start = 0
            for line in f.readlines():
                line_data = line.strip()
                if line_data == '':
                    continue

                try:
                    config_arr = line_data.split("\t")
                    col_name = config_arr[0]
                    result_type = config_arr[2]
                    result_parse_type = config_arr[6]
                    result_parse = config_arr[7]
                    is_drop = int(config_arr[8])
                    feature_name = config_arr[9]
                    is_attention = int(config_arr[10])

                    if is_drop == 1:
                        continue

                    if result_type == 'vector' or result_type == 'vec':
                        if col_name in vec_name_list:
                            vector_size += 200
                        else:
                            print("%s is error" % line)
                            exit(-1)

                    elif result_type == 'arr':
                        if result_parse_type == 'top' or result_parse_type == 'top_arr':
                            top_n = int(result_parse.strip().split("=")[1])
                            size = 1
                        elif result_parse_type == 'top_multi':
                            parse_arr = result_parse.split(";")
                            top_n = int(parse_arr[0].split("=")[1])
                            size = int(parse_arr[1].split("=")[1])
                        else:
                            print("%s is error" % line)
                            exit(-1)

                        if alg_name not in no_pool_alg:
                            index_end = index_start + top_n * size
                            index_range = [index_start, index_end, feature_name, col_name]
                            index_start = index_end
                            if is_attention == 1 and alg_name in attention_alg:
                                attention_feats_size += top_n * size
                                attention_cate_field.append(index_range)
                            else:
                                multi_cate_field += 1
                                multi_feats_size += top_n * size
                                multi_cate_range.append(index_range)
                        else:
                            cate_size += top_n * size

                    elif result_type == 'string':
                        cate_index_name = [cate_size, feature_name, col_name]
                        cate_range.append(cate_index_name)
                        cate_size += 1
                    elif result_type == 'float':
                        cont_size += 1
                    else:
                        print("%s is error!!!" % line_data)
                except Exception as e:
                    print("-----------feat_conf is Error!!!!-----------")
                    print(e)
                    print(line_data)
                    exit(-1)

    # get attention range
    for attention_cate in attention_cate_field:
        cate_match = False
        for cate in cate_range:
            if attention_cate[-2] == cate[-2] and cate[-1][:4] == 'item':
                match_tuple = (0, attention_cate[-2], cate[0], (attention_cate[0], attention_cate[1]))
                attention_range.append(match_tuple)
                cate_match = True
                break
        if not cate_match:
            for m_c in multi_cate_range:
                if attention_cate[-2] == m_c[-2] and m_c[-1][:4] == 'item':
                    match_tuple = (1, m_c[-2], (m_c[0], m_c[1]), (attention_cate[0], attention_cate[1]))
                    attention_range.append(match_tuple)
                    break

    return cont_size, vector_size, cate_size, multi_feats_size, multi_cate_range, attention_feats_size, attention_range


def fea_identify(xfea_conf_dir):
    """
    1. Read the xfea features_list conf and identify the xfea index range of features
    2. 为了支持 multi_hot，严格要求 multi_hot 代表的特征放在特征配置的最后连续几个
    3. libsvm 的索引位置严格按照 特征配置文件的 特征顺序
    Return: field_size, feature_size, col_names, feature_values, feat_name-map-position_range, multi_feats_range
    """
    multi_flag = -1
    with open(xfea_conf_dir, 'r') as xfea_conf:
        fea_size = {}
        fea_dict = {}
        fea_slot = {}
        multi_feats_range = []
        start_pos = 0

        for line in xfea_conf:
            line = line.strip()
            if not line:
                continue
            if line[0] == '#' or line[0] == '//':
                continue
            tokens = [x.split('=') for x in line.strip().split(';')]
            value_size = 1
            fea_name = ''
            fea_class = ''
            for ele in tokens:
                if 'name' == ele[0]:
                    fea_name = ele[1]
                if 'slot' == ele[0]:
                    slot = int(ele[1])
                if 'feat_values' == ele[0]:
                    value_size = len(ele[1].split(','))
                if 'hash_range_max' == ele[0]:
                    value_size = int(ele[1])
                if 'class' == ele[0]:
                    fea_class = ele[1]

            if fea_name != '' and fea_class != '':
                if fea_class == 'S_multi' and multi_flag < 0:
                    multi_flag = start_pos
                elif multi_flag > 0 and fea_class != 'S_multi':
                    raise Exception("multi_hot feature must be in the last")
                fea_slot[fea_name] = slot
                fea_size[fea_name] = value_size
                fea_dict[fea_name] = (start_pos, start_pos + value_size - 1)
                if fea_class == 'S_multi':
                    multi_feats_range.append((start_pos, start_pos + value_size - 1))
                start_pos += value_size

    cols = [_[0] for _ in sorted(fea_slot.items(), key=lambda item: item[1])]

    return len(cols), start_pos + 1, cols, multi_flag, fea_size, fea_dict, multi_feats_range


def set_dist_env(flags_obj):
    if flags_obj.dist_mode == 1:  # 本地分布式测试模式1 chief, 1 ps, 1 evaluator
        ps_hosts = flags_obj.ps_hosts.split(',')
        chief_hosts = flags_obj.chief_hosts.split(',')
        task_index = flags_obj.task_index
        job_name = flags_obj.job_name
        print('ps_host', ps_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # 无worker参数
        tf_config = {
            'cluster': {'chief': chief_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index}
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
    elif flags_obj.dist_mode == 2:  # 集群分布式模式
        ps_hosts = flags_obj.ps_hosts.split(',')
        worker_hosts = flags_obj.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1]  # get first worker as chief
        worker_hosts = worker_hosts[2:]  # the rest as worker
        task_index = flags_obj.task_index
        job_name = flags_obj.job_name
        print('ps_host', ps_hosts)
        print('worker_host', worker_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # use #worker=0 as chief
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # use #worker=1 as evaluator
        if job_name == "worker" and task_index == 1:
            job_name = 'evaluator'
            task_index = 0
        # the others as worker
        if job_name == "worker" and task_index > 1:
            task_index -= 2

        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index}
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
