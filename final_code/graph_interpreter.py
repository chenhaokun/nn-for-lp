from sklearn.metrics import roc_auc_score
from scipy.sparse.linalg import spsolve
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from data_divider import *
from sklearn import metrics
from scipy import sparse
import tensorflow as tf
import xgboost as xgb
import numpy as np
import pylab as pl
import pickle
import random
import math
import os

# use sigmoid_cross_entropy_with_logits
def mf_with_sigmoid(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter='\t')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter='\t')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.truncated_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))
    q = tf.Variable(tf.truncated_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))
    # wp = tf.Variable(tf.truncated_normal([params['embedding_size']], mean=0, stddev=0.01))
    # wq = tf.Variable(tf.truncated_normal([params['embedding_size']], mean=0, stddev=0.01))
    # bp = tf.Variable(tf.zeros([params['node_num']]))
    # bq = tf.Variable(tf.zeros([params['node_num']]))

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    # b1 = tf.nn.embedding_lookup(bp, u1s)
    # b2 = tf.nn.embedding_lookup(bq, u2s)
    dot_e = e1 * e2
    # t1 = e1 * wp
    # t2 = e2 * wq

    ys_pre = tf.reduce_sum(dot_e, 1)# + tf.reduce_sum(t1, 1) + tf.reduce_sum(t2, 1)
    # ys_pre = b1 + b2

    # target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))
    target_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(ys_pre, ys, params['train_np_rate']))
    loss = target_loss + params['beta'] * (tf.reduce_mean(tf.square(e1) + tf.square(e2)))

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        data_size = dtrain.shape[0]
        np.random.shuffle(dtrain)
        np.random.shuffle(dtest)
        print 'training mf with sigmoid...'
        rmse_v, loss_v, target_loss_v, ys_v, ys_pre_v, accuracy_v = sess.run(
            [rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
            feed_dict={u1s: dtrain[0:800000, 0],
                       u2s: dtrain[0:800000, 1],
                       ys: np.float32(dtrain[0:800000, 2])})
        t_rmse_v, t_loss_v, t_target_loss_v, t_ys_v, t_ys_pre_v, t_accuracy_v = sess.run(
            [rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
            feed_dict={u1s: dtest[0:800000, 0],
                       u2s: dtest[0:800000, 1],
                       ys: np.float32(dtest[0:800000, 2])})
        print '----round%2d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (
            0, rmse_v, loss_v, target_loss_v, get_auc(ys_v, ys_pre_v), accuracy_v, t_rmse_v, t_loss_v, t_target_loss_v,
            get_auc(t_ys_v, t_ys_pre_v), t_accuracy_v)
        stop_count = 0
        pre_auc = 0.0
        max_auc = 0.0
        for i in range(params['round']):
            for j in range(data_size / params['batch_size']):
                start = params['batch_size'] * j
                end = params['batch_size'] * (j + 1)
                feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
                sess.run(train_step, feed_dict)
            np.random.shuffle(dtrain)
            np.random.shuffle(dtest)
            rmse_v, loss_v, target_loss_v, ys_v, ys_pre_v, accuracy_v = sess.run(
                [rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
                feed_dict={u1s: dtrain[0:800000, 0],
                           u2s: dtrain[0:800000, 1],
                           ys: np.float32(dtrain[0:800000, 2])})
            t_rmse_v, t_loss_v, t_target_loss_v, t_ys_v, t_ys_pre_v, t_accuracy_v = sess.run(
                [rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
                feed_dict={u1s: dtest[0:800000, 0], u2s: dtest[0:800000, 1], ys: np.float32(dtest[0:800000, 2])})
            print '----round%2d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (
                i + 1, rmse_v, loss_v, target_loss_v, get_auc(ys_v, ys_pre_v), accuracy_v, t_rmse_v, t_loss_v,
                t_target_loss_v,
                get_auc(t_ys_v, t_ys_pre_v), t_accuracy_v)
            cur_auc = get_auc(t_ys_v, t_ys_pre_v)
            if cur_auc > max_auc:
                max_auc = cur_auc
            if (cur_auc - pre_auc) < 0.0001:
                stop_count += 1
            pre_auc = cur_auc
            if stop_count == 2:
                if params['store_test_result']:
                    p_v, q_v = sess.run([p, q], feed_dict={})
                    store_dict = {'p': list(p_v), 'q': list(q_v), 'u1s': list(dtest[0:800000, 0]), 'u2s': list(dtest[0:800000, 1]), 'ys': list(t_ys_v), 'ys_pre': list(t_ys_pre_v)}
                    store_obj(store_dict, params['mf_test_result_file_path'])
                break
        return max_auc

def cn_lr(params):
    neighbor_set_list = load_obj(params['neighbor_set_list_file_path'])
    if neighbor_set_list == -1:
        print 'no neighbor set list file exits'
        return

    with open(params['dtrain_file_path'], 'r') as dtrain_file, open(params['xgb_dtrain_file_path'], 'w') as xgb_dtrain_file:
        train_libsvm_list = []
        for line in dtrain_file:
            items = line[0:-1].split('\t')
            i = int(items[0]) - 1
            j = int(items[1]) - 1
            l = int(items[2])
            cn = len(neighbor_set_list[i][1] & neighbor_set_list[j][0])
            newline = '%d 1:%d\n' % (l, cn)
            train_libsvm_list.append(newline)
        xgb_dtrain_file.writelines(train_libsvm_list)

    with open(params['dtest_file_path'], 'r') as dtest_file, open(params['xgb_dtest_file_path'], 'w') as xgb_dtest_file:
        test_libsvm_list = []
        for line in dtest_file:
            items = line[0:-1].split('\t')
            i = int(items[0]) - 1
            j = int(items[1]) - 1
            l = int(items[2])
            cn = len(neighbor_set_list[i][1] & neighbor_set_list[j][0])
            newline = '%d 1:%d\n' % (l, cn)
            test_libsvm_list.append(newline)
        xgb_dtest_file.writelines(test_libsvm_list)

    xgboost(params['xgb_dtrain_file_path'], params['xgb_dtest_file_path'], params['xgb_params'], params['xgb_round'])

def aa_lr(params):
    neighbor_set_list = load_obj(params['neighbor_set_list_file_path'])
    if neighbor_set_list == -1:
        print 'no neighbor set list file exits'
        return

    with open(params['dtrain_file_path'], 'r') as dtrain_file, open(params['xgb_dtrain_file_path'], 'w') as xgb_dtrain_file:
        train_libsvm_list = []
        for line in dtrain_file:
            items = line[0:-1].split('\t')
            i = int(items[0]) - 1
            j = int(items[1]) - 1
            l = int(items[2])
            inter_set = neighbor_set_list[i][1] & neighbor_set_list[j][0]
            aa = np.sum([1.0 / math.log(len(neighbor_set_list[i][1])+2) for i in inter_set])
            # aa = np.sum([1.0 / (len(neighbor_set_list[i])+1) for i in inter_set])
            newline = '%d 1:%d\n' % (l, aa)
            train_libsvm_list.append(newline)
        xgb_dtrain_file.writelines(train_libsvm_list)

    with open(params['dtest_file_path'], 'r') as dtest_file, open(params['xgb_dtest_file_path'], 'w') as xgb_dtest_file:
        test_libsvm_list = []
        for line in dtest_file:
            items = line[0:-1].split('\t')
            i = int(items[0]) - 1
            j = int(items[1]) - 1
            l = int(items[2])
            inter_set = neighbor_set_list[i][1] & neighbor_set_list[j][0]
            aa = np.sum([1.0 / math.log(len(neighbor_set_list[i][1])+2) for i in inter_set])
            newline = '%d 1:%f\n' % (l, aa)
            test_libsvm_list.append(newline)
        xgb_dtest_file.writelines(test_libsvm_list)

    xgboost(params['xgb_dtrain_file_path'], params['xgb_dtest_file_path'], params['xgb_params'], params['xgb_round'])

def ra_lr(params):
    neighbor_set_list = load_obj(params['neighbor_set_list_file_path'])
    if neighbor_set_list == -1:
        print 'no neighbor set list file exits'
        return

    with open(params['dtrain_file_path'], 'r') as dtrain_file, open(params['xgb_dtrain_file_path'], 'w') as xgb_dtrain_file:
        train_libsvm_list = []
        for line in dtrain_file:
            items = line[0:-1].split('\t')
            i = int(items[0]) - 1
            j = int(items[1]) - 1
            l = int(items[2])
            inter_set = neighbor_set_list[i][1] & neighbor_set_list[j][0]
            ra = np.sum([1.0 / len(neighbor_set_list[h][1]) for h in inter_set])
            newline = '%d 1:%d\n' % (l, ra)
            train_libsvm_list.append(newline)
        xgb_dtrain_file.writelines(train_libsvm_list)

    with open(params['dtest_file_path'], 'r') as dtest_file, open(params['xgb_dtest_file_path'], 'w') as xgb_dtest_file:
        test_libsvm_list = []
        for line in dtest_file:
            items = line[0:-1].split('\t')
            i = int(items[0]) - 1
            j = int(items[1]) - 1
            l = int(items[2])
            inter_set = neighbor_set_list[i][1] & neighbor_set_list[j][0]
            ra = np.sum([1.0 / len(neighbor_set_list[h][1]) for h in inter_set])
            newline = '%d 1:%f\n' % (l, ra)
            test_libsvm_list.append(newline)
        xgb_dtest_file.writelines(test_libsvm_list)

    xgboost(params['xgb_dtrain_file_path'], params['xgb_dtest_file_path'], params['xgb_params'], params['xgb_round'])

def katz_lr(params):
    katz_matrix = load_obj(params['katz_matrix_file_path'])
    if not sparse.issparse(katz_matrix):
        print 'fail to load katz matrix'
        return

    count=0
    with open(params['dtrain_file_path'], 'r') as dtrain_file, open(params['xgb_dtrain_file_path'], 'w') as xgb_dtrain_file:
        train_libsvm_list = []
        for line in dtrain_file:
            items = line[0:-1].split('\t')
            i = int(items[0]) - 1
            j = int(items[1]) - 1
            l = int(items[2])
            katz = katz_matrix[i, j]
            newline = '%d 1:%f\n' % (l, katz)
            train_libsvm_list.append(newline)
        xgb_dtrain_file.writelines(train_libsvm_list)

    with open(params['dtest_file_path'], 'r') as dtest_file, open(params['xgb_dtest_file_path'], 'w') as xgb_dtest_file:
        test_libsvm_list = []
        for line in dtest_file:
            items = line[0:-1].split('\t')
            i = int(items[0]) - 1
            j = int(items[1]) - 1
            l = int(items[2])
            katz = katz_matrix[i, j]
            newline = '%d 1:%f\n' % (l, katz)
            test_libsvm_list.append(newline)
        xgb_dtest_file.writelines(test_libsvm_list)

    xgboost(params['xgb_dtrain_file_path'], params['xgb_dtest_file_path'], params['xgb_params'], params['xgb_round'])

def rwr_lr(params):
    rwr_matrix = load_obj(params['rwr_matrix_file_path'])
    if not sparse.issparse(rwr_matrix):
        print 'fail to load rwr matrix'
        return

    count=0
    with open(params['dtrain_file_path'], 'r') as dtrain_file, open(params['xgb_dtrain_file_path'], 'w') as xgb_dtrain_file:
        train_libsvm_list = []
        for line in dtrain_file:
            items = line[0:-1].split('\t')
            i = int(items[0]) - 1
            j = int(items[1]) - 1
            l = int(items[2])
            rwr_v = rwr_matrix[i, j]
            newline = '%d 1:%f\n' % (l, rwr_v)
            train_libsvm_list.append(newline)
        xgb_dtrain_file.writelines(train_libsvm_list)

    with open(params['dtest_file_path'], 'r') as dtest_file, open(params['xgb_dtest_file_path'], 'w') as xgb_dtest_file:
        test_libsvm_list = []
        for line in dtest_file:
            items = line[0:-1].split('\t')
            i = int(items[0]) - 1
            j = int(items[1]) - 1
            l = int(items[2])
            rwr_v = rwr_matrix[i, j]
            newline = '%d 1:%f\n' % (l, rwr_v)
            test_libsvm_list.append(newline)
        xgb_dtest_file.writelines(test_libsvm_list)

    xgboost(params['xgb_dtrain_file_path'], params['xgb_dtest_file_path'], params['xgb_params'], params['xgb_round'])

def xgboost(train_data_file_path, test_data_file_path, param, num_round):
    dtrain = xgb.DMatrix(train_data_file_path)
    dtest = xgb.DMatrix(test_data_file_path)
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    plst = param.items()
    bst = xgb.train(plst, dtrain, num_round, watchlist)
    ypred = bst.predict(dtest)
    ytrue = dtest.get_label()
    print 'xgboost test auc: %f' % get_auc(ytrue, ypred)

# pnn with outer product
def pnn(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter='\t')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter='\t')
    dtest -= [1, 1, 0]

    embedding = tf.Variable(
        tf.random_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))

    # i_biases = tf.Variable(tf.random_normal([params['node_num']], mean=0, stddev=0.01))
    # o_biases = tf.Variable(tf.random_normal([params['node_num']], mean=0, stddev=0.01))

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(embedding, u1s)
    e2 = tf.nn.embedding_lookup(embedding, u2s)

    # u1_biases = tf.nn.embedding_lookup(i_biases, u1s)
    # u2_biases = tf.nn.embedding_lookup(o_biases, u2s)

    z = tf.concat(1, [e1, e2])
    # z = tf.concat([e1, e2], 1)
    p = tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)),
                   [-1, params['embedding_size'] * params['embedding_size']])

    h0 = tf.concat(1, [z, p])
    # h0 = tf.concat([z, p], 1)

    weight1 = tf.Variable(tf.random_normal([params['embedding_size'] * (params['embedding_size'] + 2), params['h1_size']], mean=0, stddev=0.01))
    biase1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.nn.relu(tf.matmul(h0, weight1) + biase1)

    weight2 = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0, stddev=0.01))
    biase2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.nn.relu(tf.matmul(h1, weight2) + biase2)

    weight3 = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0, stddev=0.01))
    biase3 = tf.Variable(tf.zeros([params['h3_size']]))
    h3 = tf.nn.relu(tf.matmul(h2, weight3) + biase3)

    # weight4 = tf.Variable(tf.random_normal([params['h3_size'], params['h4_size']], mean=0, stddev=0.01))
    # biase4 = tf.Variable(tf.zeros([params['h4_size']]))
    # h4 = tf.nn.relu(tf.matmul(h3, weight4) + biase4)

    # weighto = tf.Variable(tf.random_normal([params['h4_size'], 1], mean=0, stddev=0.01))
    # biaseo = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h4, weighto) + biaseo)

    weighto = tf.Variable(tf.random_normal([params['h3_size'], 1], mean=0, stddev=0.01))
    biaseo = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h3, weighto) + biaseo)
    # weighto = tf.Variable(tf.random_normal([params['embedding_size'] * (params['embedding_size'] + 2), 1], mean=0, stddev=0.01))
    # ys_pre = tf.squeeze(tf.matmul(h0, weighto) + biaseo)

    # weight_l2 =params['h3_size'] tf.reduce_sum(
    #     [tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3), tf.nn.l2_loss(weight4),
    #      tf.nn.l2_loss(weighto)])

    weight_l2 = tf.reduce_sum(
        [tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3),
         tf.nn.l2_loss(weighto)])

    weight_l2 = tf.reduce_sum(
        [tf.nn.l2_loss(embedding)])

    target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))
    # target_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(ys_pre, ys, 6))
    loss = target_loss + params['beta'] * weight_l2

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        data_size = dtrain.shape[0]
        np.random.shuffle(dtrain)
        np.random.shuffle(dtest)
        print 'training pnn1...'
        rmse_v, loss_v, target_loss_v, ys_v, ys_pre_v, accuracy_v = sess.run(
            [rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
            feed_dict={u1s: dtrain[0:800000, 0], u2s: dtrain[0:800000, 1],
                       ys: np.float32(dtrain[0:800000, 2])})
        t_rmse_v, t_loss_v, t_target_loss_v, t_ys_v, t_ys_pre_v, t_accuracy_v = sess.run(
            [rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
            feed_dict={u1s: dtest[0:800000, 0], u2s: dtest[0:800000, 1],
                       ys: np.float32(dtest[0:800000, 2])})
        print '----round%2d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (
            0, rmse_v, loss_v, target_loss_v, get_auc(ys_v, ys_pre_v), accuracy_v, t_rmse_v, t_loss_v,
            t_target_loss_v,
            get_auc(t_ys_v, t_ys_pre_v), t_accuracy_v)
        stop_count = 0
        pre_auc = 0.0
        max_auc = 0.0
        store_dict = {}
        model_dict = {}
        for i in range(params['round']):
            for j in range(data_size / params['batch_size']):
                start = params['batch_size'] * j
                end = params['batch_size'] * (j + 1)
                feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1],
                             ys: np.float32(dtrain[start:end, 2])}
                sess.run(train_step, feed_dict)
            np.random.shuffle(dtrain)
            np.random.shuffle(dtest)
            rmse_v, loss_v, target_loss_v, ys_v, ys_pre_v, accuracy_v = sess.run(
                [rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
                feed_dict={u1s: dtrain[0:800000, 0], u2s: dtrain[0:800000, 1],
                           ys: np.float32(dtrain[0:800000, 2])})
            t_rmse_v, t_loss_v, t_target_loss_v, t_ys_v, t_ys_pre_v, t_accuracy_v = sess.run(
                [rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
                feed_dict={u1s: dtest[0:800000, 0], u2s: dtest[0:800000, 1],
                           ys: np.float32(dtest[0:800000, 2])})
            print '----round%2d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (
                i + 1, rmse_v, loss_v, target_loss_v, get_auc(ys_v, ys_pre_v), accuracy_v, t_rmse_v,
                t_loss_v,
                t_target_loss_v, get_auc(t_ys_v, t_ys_pre_v), t_accuracy_v)
            cur_auc = get_auc(t_ys_v, t_ys_pre_v)
            if cur_auc > max_auc:
                weight1_v, biase1_v, weight2_v, biase2_v, weight3_v, biase3_v, weighto_v, biaseo_v, embedding_v = sess.run(
                    [weight1, biase1, weight2, biase2, weight3, biase3, weighto, biaseo,
                     embedding],
                    feed_dict={u1s: dtrain[0:0, 0], u2s: dtrain[0:0, 1], ys: np.float32(dtrain[0:0, 2])})
                store_dict = {'embedding': list(embedding_v), 'u1s': list(dtest[0:800000, 0]), 'u2s': list(dtest[0:800000, 1]), 'ys': list(t_ys_v),
                              'ys_pre': list(t_ys_pre_v)}
                model_dict = {'weight1_v': weight1_v, 'biase1_v': biase1_v, 'weight2_v': weight2_v,
                             'biase2_v': biase2_v,
                             'weight3_v': weight3_v, 'biase3_v': biase3_v, 'weighto_v': weighto_v, 'biaseo_v': biaseo_v,
                             'embedding_v': embedding_v}
                max_auc = cur_auc
            if (cur_auc - pre_auc) < 0.0001:
                stop_count += 1
            pre_auc = cur_auc
            if stop_count == 50 or i == (params['round'] - 1):
                if params['store_test_result']:
                    store_obj(store_dict, params['pnn1_test_result_file_path'])
                break
        store_obj(model_dict, params['model_save_path'])
        print 'pnn1 completed'
        return max_auc

# train two pnn, one for link prediction with initialization, one for tow-lop link prediction for assisting, train alternately
def pnn_with_ann(params):
    pre_model = load_obj(params['pre_model_save_path'])
    embedding = tf.Variable(pre_model['embedding_v'])
    # i_biases = tf.Variable(pre_model['i_biases_v'])
    # o_biases = tf.Variable(pre_model['o_biases_v'])

    dtrain_a = np.loadtxt(params['dtrain_a_file_path'], delimiter='\t')
    dtrain_a -= [1, 1, 0]
    dtest_a = np.loadtxt(params['dtest_a_file_path'], delimiter='\t')
    dtest_a -= [1, 1, 0]

    u1s_a = tf.placeholder(tf.int32, shape=[None])
    u2s_a = tf.placeholder(tf.int32, shape=[None])
    ys_a = tf.placeholder(tf.float32, shape=[None])

    e1_a = tf.nn.embedding_lookup(embedding, u1s_a)
    e2_a = tf.nn.embedding_lookup(embedding, u2s_a)

    # u1_biases = tf.nn.embedding_lookup(i_biases, u1s_a)
    # u2_biases = tf.nn.embedding_lookup(o_biases, u2s_a)

    z_a = tf.concat(1, [e1_a, e2_a])
    # z_a = tf.concat([e1_a, e2_a], 1)
    p_a = tf.reshape(tf.batch_matmul(tf.expand_dims(e1_a, 2), tf.expand_dims(e2_a, 1)),
                     [-1, params['embedding_size'] * params['embedding_size']])

    h0_a = tf.concat(1, [z_a, p_a])
    # h0_a = tf.concat([z_a, p_a], 1)

    weight1_a = tf.Variable(pre_model['weight1_v'])
    biase1_a = tf.Variable(pre_model['biase1_v'])
    h1_a = tf.nn.relu(tf.matmul(h0_a, weight1_a) + biase1_a)

    weight2_a = tf.Variable(pre_model['weight2_v'])
    biase2_a = tf.Variable(pre_model['biase2_v'])
    h2_a = tf.nn.relu(tf.matmul(h1_a, weight2_a) + biase2_a)

    weight3_a = tf.Variable(pre_model['weight3_v'])
    biase3_a = tf.Variable(pre_model['biase3_v'])
    h3_a = tf.nn.relu(tf.matmul(h2_a, weight3_a) + biase3_a)

    # weight4_a = tf.Variable(pre_model['weight4_v'])
    # biase4_a = tf.Variable(pre_model['biase4_v'])
    # h4_a = tf.nn.relu(tf.matmul(h3_a, weight4_a) + biase4_a)

    # weighto_a = tf.Variable(pre_model['weighto_v'])
    # biaseo_a = tf.Variable(pre_model['biaseo_v'])
    # ys_pre_a = tf.squeeze(tf.matmul(h4_a, weighto_a) + biaseo_a)

    weighto_a = tf.Variable(pre_model['weighto_v'])
    biaseo_a = tf.Variable(pre_model['biaseo_v'])
    ys_pre_a = tf.squeeze(tf.matmul(h3_a, weighto_a) + biaseo_a)

    # weight_l2_a = tf.reduce_sum(
    #     [tf.nn.l2_loss(weight1_a), tf.nn.l2_loss(weight2_a), tf.nn.l2_loss(weight3_a),
    #      tf.nn.l2_loss(weight4_a),
    #      tf.nn.l2_loss(weighto_a)])

    weight_l2_a = tf.reduce_sum(
        [tf.nn.l2_loss(weight1_a), tf.nn.l2_loss(weight2_a), tf.nn.l2_loss(weight3_a), tf.nn.l2_loss(weighto_a)])
    weight_l2_a = tf.reduce_sum(
        [tf.nn.l2_loss(embedding)])

    target_loss_a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre_a, ys_a))
    loss_a = target_loss_a + params['beta1'] * weight_l2_a

    train_step_a = tf.train.AdamOptimizer(params['learning_rate1']).minimize(loss_a)

    p_list = np.loadtxt(params['dtrain_b_file_path'], delimiter='\t')
    p_train = np.ones([len(p_list), 3])
    p_train[:, 0:2] = p_list
    p_set = set()
    n_set = set()
    p_num = p_list.shape[0]
    n_num = p_num * params['hop2_np_rate']
    for i in p_list:
        p_set.add('%d %d'%(i[0], i[1]))

    while (len(n_set) < n_num):
        a = random.randint(1, params['node_num'])
        b = random.randint(1, params['node_num'])
        line = '%d %d' % (a, b)
        if a != b and line not in p_set and line not in n_set:
            n_set.add(line)

    n_list = [[int(s) for s in line.split(' ')] for line in n_set]
    n_train = np.zeros([len(n_list), 3])
    n_train[:,0:2] = n_list
    dtrain_b = np.concatenate((p_train, n_train), axis=0)
    dtrain_b -= [1, 1, 0]

    u1s_b = tf.placeholder(tf.int32, shape=[None])
    u2s_b = tf.placeholder(tf.int32, shape=[None])
    ys_b = tf.placeholder(tf.float32, shape=[None])

    e1_b = tf.nn.embedding_lookup(embedding, u1s_b)
    e2_b = tf.nn.embedding_lookup(embedding, u2s_b)

    z_b = tf.concat(1, [e1_b, e2_b])
    # z_b = tf.concat([e1_b, e2_b], 1)
    p_b = tf.reshape(tf.batch_matmul(tf.expand_dims(e1_b, 2), tf.expand_dims(e2_b, 1)),
                     [-1, params['embedding_size'] * params['embedding_size']])

    h0_b = tf.concat(1, [z_b, p_b])
    # h0_b = tf.concat([z_b, p_b], 1)

    weight1_b = tf.Variable(
        tf.random_normal([params['embedding_size'] * (params['embedding_size'] + 2), params['h1_size']],
                         mean=0,
                         stddev=0.01))
    biase1_b = tf.Variable(tf.zeros([params['h1_size']]))
    h1_b = tf.nn.relu(tf.matmul(h0_b, weight1_b) + biase1_b)

    weight2_b = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0, stddev=0.01))
    biase2_b = tf.Variable(tf.zeros([params['h2_size']]))
    h2_b = tf.nn.relu(tf.matmul(h1_b, weight2_b) + biase2_b)

    weight3_b = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0, stddev=0.01))
    biase3_b = tf.Variable(tf.zeros([params['h3_size']]))
    h3_b = tf.nn.relu(tf.matmul(h2_b, weight3_b) + biase3_b)

    # weight4_b = tf.Variable(tf.random_normal([params['h3_size'], params['h4_size']], mean=0, stddev=0.01))
    # biase4_b = tf.Variable(tf.zeros([params['h4_size']]))
    # h4_b = tf.nn.relu(tf.matmul(h3_b, weight4_b) + biase4_b)

    # weighto_b = tf.Variable(tf.random_normal([params['h4_size'], 1], mean=0, stddev=0.01))
    # biaseo_b = tf.Variable(tf.zeros([1]))
    # ys_pre_b = tf.squeeze(tf.matmul(h4_b, weighto_b) + biaseo_b)

    weighto_b = tf.Variable(tf.random_normal([params['h3_size'], 1], mean=0, stddev=0.01))
    biaseo_b = tf.Variable(tf.zeros([1]))
    ys_pre_b = tf.squeeze(tf.matmul(h3_b, weighto_b) + biaseo_b)

    # weight_l2_b = tf.reduce_sum(
    #     [tf.nn.l2_loss(weight1_b), tf.nn.l2_loss(weight2_b), tf.nn.l2_loss(weight3_b),
    #      tf.nn.l2_loss(weight4_b),
    #      tf.nn.l2_loss(weighto_b)])

    weight_l2_b = tf.reduce_sum(
        [tf.nn.l2_loss(weight1_b), tf.nn.l2_loss(weight2_b), tf.nn.l2_loss(weight3_b), tf.nn.l2_loss(weighto_b)])
    weight_l2_b = tf.reduce_sum(
        [tf.nn.l2_loss(embedding)])

    target_loss_b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre_b, ys_b))
    loss_b = target_loss_b + params['beta2'] * weight_l2_b

    train_step_b = tf.train.AdamOptimizer(params['learning_rate2']).minimize(loss_b)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print 'training pnn2...'

        data_size_a = dtrain_a.shape[0]
        np.random.shuffle(dtrain_a)
        np.random.shuffle(dtest_a)
        loss_a_v, target_loss_a_v, ys_a_v, ys_pre_a_v = sess.run(
            [loss_a, target_loss_a, ys_a, tf.sigmoid(ys_pre_a)],
            feed_dict={u1s_a: dtrain_a[0:800000, 0],
                       u2s_a: dtrain_a[0:800000, 1],
                       ys_a: np.float32(dtrain_a[0:800000, 2])})
        t_loss_a_v, t_target_loss_a_v, t_ys_a_v, t_ys_pre_a_v = sess.run(
            [loss_a, target_loss_a, ys_a, tf.sigmoid(ys_pre_a)],
            feed_dict={u1s_a: dtest_a[0:800000, 0], u2s_a: dtest_a[0:800000, 1],
                       ys_a: np.float32(dtest_a[0:800000, 2])})

        data_size_b = dtrain_b.shape[0]
        np.random.shuffle(dtrain_b)
        loss_b_v, target_loss_b_v, ys_b_v, ys_pre_b_v = sess.run(
            [loss_b, target_loss_b, ys_b, tf.sigmoid(ys_pre_b)],
            feed_dict={u1s_b: dtrain_b[0:800000, 0],
                       u2s_b: dtrain_b[0:800000, 1],
                       ys_b: np.float32(dtrain_b[0:800000, 2])})

        print '----round%2d: loss1: %f, target_loss1: %f, auc1: %f, t_loss1: %f, t_target_loss1: %f, t_auc1: %f ----loss2: %f, target_loss2: %f, auc2: %f' % (
            0, loss_a_v, target_loss_a_v, get_auc(ys_a_v, ys_pre_a_v), t_loss_a_v, t_target_loss_a_v,
            get_auc(t_ys_a_v, t_ys_pre_a_v), loss_b_v, target_loss_b_v, get_auc(ys_b_v, ys_pre_b_v))

        steps_of_round_a = data_size_a / params['batch_size']
        steps_of_round_b = data_size_b / params['batch_size']
        stop_count = 0
        pre_auc = 0.0
        max_auc = 0.0
        for i in range(params['round']):
            j = 0
            while (j < steps_of_round_b):
                start = params['batch_size'] * j
                end = params['batch_size'] * (j + 1)
                feed_dict = {u1s_b: dtrain_b[start:end, 0], u2s_b: dtrain_b[start:end, 1],
                             ys_b: np.float32(dtrain_b[start:end, 2])}
                sess.run(train_step_b, feed_dict)
                j += 1
            loss_b_v, target_loss_b_v, ys_b_v, ys_pre_b_v = sess.run(
                [loss_b, target_loss_b, ys_b, tf.sigmoid(ys_pre_b)],
                feed_dict={u1s_b: dtrain_b[0:800000, 0], u2s_b: dtrain_b[0:800000, 1],
                           ys_b: np.float32(dtrain_b[0:800000, 2])})

            j = 0
            while (j < steps_of_round_a):
                start = params['batch_size'] * j
                end = params['batch_size'] * (j + 1)
                feed_dict = {u1s_a: dtrain_a[start:end, 0], u2s_a: dtrain_a[start:end, 1],
                             ys_a: np.float32(dtrain_a[start:end, 2])}
                sess.run(train_step_a, feed_dict)
                j += 1

            np.random.shuffle(dtrain_a)
            np.random.shuffle(dtest_a)
            np.random.shuffle(dtrain_b)
            loss_a_v, target_loss_a_v, ys_a_v, ys_pre_a_v = sess.run(
                [loss_a, target_loss_a, ys_a, tf.sigmoid(ys_pre_a)],
                feed_dict={u1s_a: dtrain_a[0:800000, 0], u2s_a: dtrain_a[0:800000, 1],
                           ys_a: np.float32(dtrain_a[0:800000, 2])})
            t_loss_a_v, t_target_loss_a_v, t_ys_a_v, t_ys_pre_a_v = sess.run(
                [loss_a, target_loss_a, ys_a, tf.sigmoid(ys_pre_a)],
                feed_dict={u1s_a: dtest_a[0:800000, 0], u2s_a: dtest_a[0:800000, 1],
                           ys_a: np.float32(dtest_a[0:800000, 2])})

            print '----round%2d: loss1: %f, target_loss1: %f, auc1: %f, t_loss1: %f, t_target_loss1: %f, t_auc1: %f ----loss2: %f, target_loss2: %f, auc2: %f' % (
                i + 1, loss_a_v, target_loss_a_v, get_auc(ys_a_v, ys_pre_a_v), t_loss_a_v,
                t_target_loss_a_v,
                get_auc(t_ys_a_v, t_ys_pre_a_v), loss_b_v, target_loss_b_v, get_auc(ys_b_v, ys_pre_b_v))

            cur_auc = get_auc(t_ys_a_v, t_ys_pre_a_v)
            if cur_auc > max_auc:
                embedding_v = sess.run([embedding], feed_dict={})
                store_dict = {'embedding': list(embedding_v), 'u1s': list(dtest_a[0:800000, 0]), 'u2s': list(dtest_a[0:800000, 1]), 'ys': list(t_ys_a_v), 'ys_pre': list(t_ys_pre_a_v)}
                max_auc = cur_auc
            if (cur_auc - pre_auc) < 0.0001:
                stop_count += 1
            pre_auc = cur_auc
            if stop_count == 20 or i==(params['round']-1):
                if params['store_test_result']:
                    store_obj(store_dict, params['pnn2_test_result_file_path'])
                break
        return max_auc

def get_max_node_num(source_file_path):
    print('starting get_max_node_num')

    with open(source_file_path, 'r') as source_file:
        max_num = 0
        count = 0
        for line in source_file:
            items = (line[0:-1]).split('\t')
            i = int(items[0])
            j = int(items[1])
            max_num = max(max_num, i, j)
            count += 1
        print('\tmax_node_num: %d, all positive edge count: %d' % (max_num, count))
        print('get_max_node_num completed\n')
        return max_num

def get_auc(y_trues, y_pres):
    fpr, tpr, thresholds = metrics.roc_curve(np.int32(y_trues), y_pres, pos_label=1)
    return metrics.auc(fpr, tpr)

def store_obj(obj, file_path):
    with open(file_path,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_path):
    if not os.path.exists(file_path):
        return -1
    with open(file_path,'rb') as f:
        return pickle.load(f)

def base_exp(params):
    path_list = params['source_file_path'].split('/')
    dir = '/'.join(path_list[0:-1]) + '/'
    data_name = path_list[-1].split('_')[0]
    version = params['version']
    node_num = get_max_node_num(params['source_file_path'])
    if 'cn' in params['baseline_set']:
        cn_lr({'dtrain_file_path': dir + '%s_train_data_v%d' % (data_name, version),
               'dtest_file_path': dir + '%s_test_data_v%d' % (data_name, version),
               'neighbor_set_list_file_path': dir + '%s_train_neighbor_set_list_v%d' % (data_name, version),
               'xgb_dtrain_file_path': dir + '%s_xgb_cn_train_data_v%d' % (data_name, version),
               'xgb_dtest_file_path': dir + '%s_xgb_cn_test_data_v%d' % (data_name, version),
               'xgb_params': {'max_depth':6, 'eta':0.3, 'silent':0, 'objective':'binary:logistic', 'nthread':8},
               'xgb_round': 10})
    if 'aa' in params['baseline_set']:
        aa_lr({'dtrain_file_path': dir + '%s_train_data_v%d' % (data_name, version),
               'dtest_file_path': dir + '%s_test_data_v%d' % (data_name, version),
               'neighbor_set_list_file_path': dir + '%s_train_neighbor_set_list_v%d' % (data_name, version),
               'xgb_dtrain_file_path': dir + '%s_xgb_aa_train_data_v%d' % (data_name, version),
               'xgb_dtest_file_path': dir + '%s_xgb_aa_test_data_v%d' % (data_name, version),
               'xgb_params': {'max_depth':6, 'eta':0.3, 'silent':0, 'objective':'binary:logistic', 'nthread':8},
               'xgb_round': 10})
    if 'ra' in params['baseline_set']:
        ra_lr({'dtrain_file_path': dir + '%s_train_data_v%d' % (data_name, version),
               'dtest_file_path': dir + '%s_test_data_v%d' % (data_name, version),
               'neighbor_set_list_file_path': dir + '%s_train_neighbor_set_list_v%d' % (data_name, version),
               'xgb_dtrain_file_path': dir + '%s_xgb_ra_train_data_v%d' % (data_name, version),
               'xgb_dtest_file_path': dir + '%s_xgb_ra_test_data_v%d' % (data_name, version),
               'xgb_params': {'max_depth':6, 'eta':0.3, 'silent':0, 'objective':'binary:logistic', 'nthread':8},
               'xgb_round': 10})
    if 'katz' in params['baseline_set']:
        katz_lr({'dtrain_file_path': dir + '%s_train_data_v%d' % (data_name, version),
               'dtest_file_path': dir + '%s_test_data_v%d' % (data_name, version),
               'katz_matrix_file_path': dir + '%s_train_katz_matrix_v%d' % (data_name, version),
               'xgb_dtrain_file_path': dir + '%s_xgb_katz_train_data_v%d' % (data_name, version),
               'xgb_dtest_file_path': dir + '%s_xgb_katz_test_data_v%d' % (data_name, version),
               'xgb_params': {'max_depth':6, 'eta':0.3, 'silent':0, 'objective':'binary:logistic', 'nthread':8},
               'xgb_round': 10})
    if 'rwr' in params['baseline_set']:
        rwr_lr({'dtrain_file_path': dir + '%s_train_data_v%d' % (data_name, version),
               'dtest_file_path': dir + '%s_test_data_v%d' % (data_name, version),
               'rwr_matrix_file_path': dir + '%s_train_rwr_matrix_v%d' % (data_name, version),
               'xgb_dtrain_file_path': dir + '%s_xgb_rwr_train_data_v%d' % (data_name, version),
               'xgb_dtest_file_path': dir + '%s_xgb_rwr_test_data_v%d' % (data_name, version),
               'xgb_params': {'max_depth':6, 'eta':0.3, 'silent':0, 'objective':'binary:logistic', 'nthread':8},
               'xgb_round': 10})
    if 'mf' in params['baseline_set']:
        mf_with_sigmoid({'dtrain_file_path': dir + '%s_train_data_v%d' % (data_name, version),
                         'dtest_file_path': dir + '%s_test_data_v%d' % (data_name, version),
                         'embedding_size': 20, 'node_num': node_num, 'train_np_rate': params['train_np_rate'],
                         'batch_size': 5000, 'round': 25, 'learning_rate': 5e-3, 'beta': 4e-1,
                         'store_test_result': False,
                         'mf_test_result_file_path': dir + '%s_mf_test_result_v%d' % (data_name, version)})

    if len(params['pnn1'])>0:
        p = params['pnn1']
        lr = 1e-2
        bt = 1e-4
        if 'learning_rate' in p and 'beta' in p:
            lr = p['learning_rate']
            bt = p['beta']
        pnn({'dtrain_file_path': dir + '%s_train_data_v%d' % (data_name, version),
             'dtest_file_path': dir + '%s_test_data_v%d' % (data_name, version),
             'model_save_path': dir + '%s_data_pnn_model_params_saver_v%d' % (data_name, version),
             'embedding_size': 20, 'node_num': node_num,
             'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
             'round': p['round'], 'learning_rate': lr, 'beta': bt, 'batch_size': 3000, #'learning_rate': 4e-3, 'beta': 4e-4, 'batch_size': 5000,
             'store_test_result': False,
             'pnn1_test_result_file_path': dir + '%s_pnn1_test_result_v%d' % (data_name, version)})

    if len(params['pnn2'])>0:
        p = params['pnn2']
        pnn_with_ann({'dtrain_a_file_path': dir + '%s_train_data_v%d' % (data_name, version),
                           'dtest_a_file_path': dir + '%s_test_data_v%d' % (data_name, version),
                           'dtrain_b_file_path': dir + '%s_hop2_train_positive_data_v%d' % (data_name, version),
                           'pre_model_save_path': dir + '%s_data_pnn_model_params_saver_v%d' % (data_name, version),
                           'embedding_size': 20, 'node_num': node_num,
                           'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
                           'batch_size': 5000, 'round': p['round'], 'learning_rate1': p['learning_rate1'], 'learning_rate2': p['learning_rate2'], 'beta1': p['beta1'], 'beta2': p['beta2'],
                           'hop2_np_rate': p['hop2_np_rate'],
                           'store_test_result': False,
                           'pnn2_test_result_file_path': dir + '%s_pnn2_test_result_v%d' % (data_name, version)
                           })