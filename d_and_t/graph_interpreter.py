from sklearn.metrics import roc_auc_score
from scipy.sparse.linalg import spsolve
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

# normal svd
def svd(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter='\t')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter='\t')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.truncated_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))
    q = tf.Variable(tf.truncated_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    # ys_pre = tf.minimum(tf.nn.relu(tf.reduce_sum(dot_e, 1)), 1.0)
    ys_pre = tf.reduce_sum(dot_e, 1)

    target_loss = tf.reduce_mean(0.5 * tf.square(ys - ys_pre))
    loss = target_loss + params['beta'] * (tf.reduce_mean(tf.square(e1) + tf.square(e2)))

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - ys_pre)))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(ys_pre, tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        data_size = dtrain.shape[0]
        np.random.shuffle(dtrain)
        np.random.shuffle(dtest)
        print 'training svd...'
        rmse_v, loss_v, target_loss_v, ys_v, ys_pre_v, accuracy_v = sess.run(
            [rmse, loss, target_loss, ys, ys_pre, accuracy],
            feed_dict={u1s: dtrain[0:800000, 0],
                       u2s: dtrain[0:800000, 1],
                       ys: np.float32(dtrain[0:800000, 2])})
        t_rmse_v, t_loss_v, t_target_loss_v, t_ys_v, t_ys_pre_v, t_accuracy_v = sess.run(
            [rmse, loss, target_loss, ys, ys_pre, accuracy],
            feed_dict={u1s: dtest[0:800000, 0],
                       u2s: dtest[0:800000, 1],
                       ys: np.float32(dtest[0:800000, 2])})
        print '----round%2d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (
            0, rmse_v, loss_v, target_loss_v, get_auc(ys_v, ys_pre_v), accuracy_v, t_rmse_v, t_loss_v, t_target_loss_v,
            get_auc(t_ys_v, t_ys_pre_v), t_accuracy_v)
        stop_count = 0
        pre_auc = 0.0
        for i in range(params['round']):
            for j in range(data_size / params['batch_size']):
                start = params['batch_size'] * j
                end = params['batch_size'] * (j + 1)
                feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
                sess.run(train_step, feed_dict)
            np.random.shuffle(dtrain)
            np.random.shuffle(dtest)
            rmse_v, loss_v, target_loss_v, ys_v, ys_pre_v, accuracy_v = sess.run(
                [rmse, loss, target_loss, ys, ys_pre, accuracy],
                feed_dict={u1s: dtrain[0:800000, 0],
                           u2s: dtrain[0:800000, 1],
                           ys: np.float32(dtrain[0:800000, 2])})
            t_rmse_v, t_loss_v, t_target_loss_v, t_ys_v, t_ys_pre_v, t_accuracy_v = sess.run(
                [rmse, loss, target_loss, ys, ys_pre, accuracy],
                feed_dict={u1s: dtest[0:800000, 0], u2s: dtest[0:800000, 1], ys: np.float32(dtest[0:800000, 2])})
            print '----round%2d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (
                i + 1, rmse_v, loss_v, target_loss_v, get_auc(ys_v, ys_pre_v), accuracy_v, t_rmse_v, t_loss_v,
                t_target_loss_v, get_auc(t_ys_v, t_ys_pre_v), t_accuracy_v)
            cur_auc = get_auc(t_ys_v, t_ys_pre_v)
            if (cur_auc - pre_auc) < 0.0001:
                stop_count += 1
            pre_auc = cur_auc
            if stop_count == 2:
                break

# use sigmoid_cross_entropy_with_logits
def mf_with_sigmoid(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter='\t')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter='\t')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.truncated_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))
    q = tf.Variable(tf.truncated_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    ys_pre = tf.reduce_sum(dot_e, 1)

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
            if (cur_auc - pre_auc) < 0.0001:
                stop_count += 1
            pre_auc = cur_auc
            if stop_count == 2:
                if params['store_test_result']:
                    store_dict = {'u1s': dtest[0:800000, 0], 'u2s': dtest[0:800000, 1], 'ys': t_ys_v, 'ys_pre': t_ys_pre_v}
                    store_obj(store_dict, params['mf_test_result_file_path'])
                break

def show_auc_curve(params):
    neighbor_set_list = load_obj(params['neighbor_set_list_file_path'])
    points = params['x_max']/params['x_step']
    x_list = range(0+params['x_step'], params['x_max']+params['x_step'], params['x_step'])
    m_list = ['mf', 'pnn1', 'pnn2']
    plot_list = []
    color_list = ['r', 'g', 'b']

    for j in range(len(m_list)):
        step_ys_list = [[] for i in range(points)]
        step_ys_pre_list = [[] for i in range(points)]
        result_dict = load_obj(params[m_list[j]+'_test_result_file_path'])
        u1s_list = result_dict['u1s']
        u2s_list = result_dict['u2s']
        ys_list = result_dict['ys']
        ys_pre_list = result_dict['ys_pre']
        for i in range(len(u1s_list)):
            i1 = len(neighbor_set_list[int(u1s_list[i])][0])
            o1 = len(neighbor_set_list[int(u1s_list[i])][1])
            i2 = len(neighbor_set_list[int(u2s_list[i])][0])
            o2 = len(neighbor_set_list[int(u2s_list[i])][1])
            neighbor_num = min(i1+o1, i2+o2)#max(o1, i2)
            if neighbor_num > 0 and neighbor_num < params['x_max']:
                index = (neighbor_num-1) / params['x_step']
                step_ys_list[index].append(ys_list[i])
                step_ys_pre_list[index].append(ys_pre_list[i])
        auc_list = []
        tmp_list1 = []
        tmp_list2 = []
        for i in range(points):
            tmp_list1 = np.concatenate((step_ys_list[i], tmp_list1))
            tmp_list2 = np.concatenate((step_ys_pre_list[i], tmp_list2))
            # tmp_list1=np.array(step_ys_list[i])
            # tmp_list2 = np.array(step_ys_pre_list[i])
            # if i==0 and j==2:
            #     print tmp_list1, tmp_list2
            auc_list.append(get_auc(tmp_list1, tmp_list2))
        print(m_list[j])
        for i in range(points):
            print(auc_list[i])
        plot_list.append(pl.plot(x_list, auc_list, color_list[j]))

    pl.title('cold start auc')  # give plot a title
    pl.xlabel('node degree')  # make axis labels
    pl.ylabel('auc')

    # pl.xlim(1, 10)  # set axis limits
    pl.ylim(0.78, 1.00)

    pl.legend(plot_list, ('mf', 'pnn1', 'pnn2'))  # make legend
    pl.show()  # show the plot on the screen

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
            # aa = np.sum([1.0 / (len(neighbor_set_list[i])+1) for i in inter_set])
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
            ra = np.sum([1.0 / len(neighbor_set_list[i][1]) for i in inter_set])
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
            ra = np.sum([1.0 / len(neighbor_set_list[i][1]) for i in inter_set])
            newline = '%d 1:%f\n' % (l, ra)
            test_libsvm_list.append(newline)
        xgb_dtest_file.writelines(test_libsvm_list)

    xgboost(params['xgb_dtrain_file_path'], params['xgb_dtest_file_path'], params['xgb_params'], params['xgb_round'])

def katz_lr(params):
    katz_matrix = load_obj(params['katz_matrix_file_path'])
    if not sparse.issparse(katz_matrix):
        print 'fail to load katz matrix'
        return
    # if not sparse.isspmatrix_dok(katz_matrix):
    #     print('to dok...')
    #     katz_matrix = katz_matrix.todok()
    #     store_obj(katz_matrix, params['katz_matrix_file_path'])

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
            # count+=1
            # if count%10000==0:
            #     print '%dw'%(count/10000)
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
    # if not sparse.isspmatrix_dok(katz_matrix):
    #     print('to dok...')
    #     katz_matrix = katz_matrix.todok()
    #     store_obj(katz_matrix, params['katz_matrix_file_path'])

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
            # count+=1
            # if count%10000==0:
            #     print '%dw'%(count/10000)
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
    # ytrue = np.int32(np.loadtxt(test_data_file_path, dtype='str', delimiter=' ')[:, 0])
    print 'xgboost test auc: %f' % get_auc(ytrue, ypred)

def fixed_emb_pnn2(params):
    pre_model = load_obj(params['pre_model_save_path'])
    embedding = tf.constant(pre_model['embedding_v'])

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter='\t')
    dtrain -= [1, 1, 0]

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(embedding, u1s)
    e2 = tf.nn.embedding_lookup(embedding, u2s)

    z = tf.concat(1, [e1, e2])
    p = tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)),
                     [-1, params['embedding_size'] * params['embedding_size']])

    h0 = tf.concat(1, [z, p])

    weight1 = tf.Variable(
        tf.random_normal([params['embedding_size'] * (params['embedding_size'] + 2), params['h1_size']],
                         mean=0,
                         stddev=0.01))
    biase1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.nn.relu(tf.matmul(h0, weight1) + biase1)

    weight2 = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0, stddev=0.01))
    biase2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.nn.relu(tf.matmul(h1, weight2) + biase2)

    weight3 = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0, stddev=0.01))
    biase3 = tf.Variable(tf.zeros([params['h3_size']]))
    h3 = tf.nn.relu(tf.matmul(h2, weight3) + biase3)

    weight4 = tf.Variable(tf.random_normal([params['h3_size'], params['h4_size']], mean=0, stddev=0.01))
    biase4 = tf.Variable(tf.zeros([params['h4_size']]))
    h4 = tf.nn.relu(tf.matmul(h3, weight4) + biase4)

    weighto = tf.Variable(tf.random_normal([params['h4_size'], 1], mean=0, stddev=0.01))
    biaseo = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h4, weighto) + biaseo)

    weight_l2 = tf.reduce_sum(
        [tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3),
         tf.nn.l2_loss(weight4),
         tf.nn.l2_loss(weighto)])

    target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))
    loss = target_loss + params['beta'] * weight_l2

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print 'training pnn2 with fixed embedding...'

        data_size = dtrain.shape[0]
        np.random.shuffle(dtrain)
        loss_v, target_loss_v, ys_v, ys_pre_v = sess.run(
            [loss, target_loss, ys, tf.sigmoid(ys_pre)],
            feed_dict={u1s: dtrain[0:800000, 0],
                       u2s: dtrain[0:800000, 1],
                       ys: np.float32(dtrain[0:800000, 2])})

        print '----round%2d: loss: %f, target_loss: %f, auc: %f' % (
            0, loss_v, target_loss_v, get_auc(ys_v, ys_pre_v))

        steps_of_round = data_size / params['batch_size']
        stop_count = 0
        pre_auc = 0.0
        for i in range(params['round']):
            j = 0
            while (j < steps_of_round):
                start = params['batch_size'] * j
                end = params['batch_size'] * (j + 1)
                feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1],
                             ys: np.float32(dtrain[start:end, 2])}
                sess.run(train_step, feed_dict)
                j += 1

            np.random.shuffle(dtrain)
            loss_v, target_loss_v, ys_v, ys_pre_v = sess.run(
                [loss, target_loss, ys, tf.sigmoid(ys_pre)],
                feed_dict={u1s: dtrain[0:800000, 0], u2s: dtrain[0:800000, 1],
                           ys: np.float32(dtrain[0:800000, 2])})

            print '----round%2d: loss: %f, target_loss: %f, auc: %f' % (
                i + 1, loss_v, target_loss_v, get_auc(ys_v, ys_pre_v))

            cur_auc = get_auc(ys_v, ys_pre_v)
            if (cur_auc - pre_auc) < 0.0001:
                stop_count += 1
            pre_auc = cur_auc
            if stop_count == 3:
                break

# pnn with outer product
def pnn_test(params):
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
    p = tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)),
                   [-1, params['embedding_size'] * params['embedding_size']])

    h0 = tf.concat(1, [z, p])

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

    # weight_l2 = tf.reduce_sum(
    #     [tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3), tf.nn.l2_loss(weight4),
    #      tf.nn.l2_loss(weighto)])

    weight_l2 = tf.reduce_sum(
        [tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3),
         tf.nn.l2_loss(weighto)])

    target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))
    # target_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(ys_pre, ys, 6))
    loss = target_loss + params['beta'] * weight_l2

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
            if (cur_auc - pre_auc) < 0.0001:
                stop_count += 1
            pre_auc = cur_auc
            if stop_count == 3 or i == (params['round'] - 1):
                if params['store_test_result']:
                    store_dict = {'u1s': dtest[0:800000, 0], 'u2s': dtest[0:800000, 1], 'ys': t_ys_v,
                                  'ys_pre': t_ys_pre_v}
                    store_obj(store_dict, params['pnn1_test_result_file_path'])
                break
        weight1_v, biase1_v, weight2_v, biase2_v, weight3_v, biase3_v, weighto_v, biaseo_v, embedding_v = sess.run(
            [weight1, biase1, weight2, biase2, weight3, biase3, weighto, biaseo,
             embedding],
            feed_dict={u1s: dtrain[0:0, 0], u2s: dtrain[0:0, 1], ys: np.float32(dtrain[0:0, 2])})
        save_dict = {'weight1_v': weight1_v, 'biase1_v': biase1_v, 'weight2_v': weight2_v,
                     'biase2_v': biase2_v,
                     'weight3_v': weight3_v, 'biase3_v': biase3_v, 'weighto_v': weighto_v, 'biaseo_v': biaseo_v, 'embedding_v': embedding_v}
        store_obj(save_dict, params['model_save_path'])
        print 'pnn1 completed'

# two embedding p and q
def pnn_test2(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter='\t')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter='\t')
    dtest -= [1, 1, 0]

    p = tf.Variable(
        tf.random_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))
    q = tf.Variable(
        tf.random_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)

    z = tf.concat(1, [e1, e2])
    tp = tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)),
                   [-1, params['embedding_size'] * params['embedding_size']])

    h0 = tf.concat(1, [z, tp])

    weight1 = tf.Variable(tf.random_normal([params['embedding_size'] * (params['embedding_size'] + 2), params['h1_size']], mean=0, stddev=0.01))
    biase1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.nn.relu(tf.matmul(h0, weight1) + biase1)

    weight2 = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0, stddev=0.01))
    biase2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.nn.relu(tf.matmul(h1, weight2) + biase2)

    weight3 = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0, stddev=0.01))
    biase3 = tf.Variable(tf.zeros([params['h3_size']]))
    h3 = tf.nn.relu(tf.matmul(h2, weight3) + biase3)

    weight4 = tf.Variable(tf.random_normal([params['h3_size'], params['h4_size']], mean=0, stddev=0.01))
    biase4 = tf.Variable(tf.zeros([params['h4_size']]))
    h4 = tf.nn.relu(tf.matmul(h3, weight4) + biase4)

    weighto = tf.Variable(tf.random_normal([params['h4_size'], 1], mean=0, stddev=0.01))
    biaseo = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h4, weighto) + biaseo)

    weight_l2 = tf.reduce_sum(
        [tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3), tf.nn.l2_loss(weight4),
         tf.nn.l2_loss(weighto)])

    target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))
    # target_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(ys_pre, ys, 6))
    loss = target_loss + params['beta'] * weight_l2

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
            if (cur_auc - pre_auc) < 0.0001:
                stop_count += 1
            pre_auc = cur_auc
            if stop_count == 3 or i == (params['round'] - 1):
                if params['store_test_result']:
                    store_dict = {'u1s': dtest[0:800000, 0], 'u2s': dtest[0:800000, 1], 'ys': t_ys_v,
                                  'ys_pre': t_ys_pre_v}
                    store_obj(store_dict, params['pnn1_test_result_file_path'])
                break
        weight1_v, biase1_v, weight2_v, biase2_v, weight3_v, biase3_v, weight4_v, biase4_v, weighto_v, biaseo_v, p_v, q_v = sess.run(
            [weight1, biase1, weight2, biase2, weight3, biase3, weight4, biase4, weighto, biaseo,
             p, q],
            feed_dict={u1s: dtrain[0:0, 0], u2s: dtrain[0:0, 1], ys: np.float32(dtrain[0:0, 2])})
        save_dict = {'weight1_v': weight1_v, 'biase1_v': biase1_v, 'weight2_v': weight2_v,
                     'biase2_v': biase2_v,
                     'weight3_v': weight3_v, 'biase3_v': biase3_v, 'weight4_v': weight4_v,
                     'biase4_v': biase4_v,
                     'weighto_v': weighto_v, 'biaseo_v': biaseo_v, 'p_v': p_v, 'q_v': q_v}
        store_obj(save_dict, params['model_save_path'])

# params exp
def pnn_test3(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter='\t')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter='\t')
    dtest -= [1, 1, 0]

    embedding = tf.Variable(tf.random_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(embedding, u1s)
    e2 = tf.nn.embedding_lookup(embedding, u2s)

    z = tf.concat(1, [e1, e2])
    p = tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)),
                   [-1, params['embedding_size'] * params['embedding_size']])

    h0 = tf.concat(1, [z, p])

    weight_list = []
    biase_list = []
    h_list = []
    h_pre = 0
    h_cur = 0
    for i in range(len(params['h_size_list'])):
        if i == 0:
            h_pre = params['embedding_size'] * (params['embedding_size'] + 2)
            h_list.append(h0)
        else:
            h_pre = params['h_size_list'][i-1]
        h_cur = params['h_size_list'][i]

        weight_list.append(tf.Variable(tf.random_normal([h_pre, h_cur], mean=0, stddev=0.01)))
        biase_list.append(tf.Variable(tf.zeros([h_cur])))
        h_list.append(tf.nn.relu(tf.matmul(h_list[-1], weight_list[-1]) + biase_list[-1]))

    weight_list.append(tf.Variable(tf.random_normal([params['h_size_list'][-1], 1], mean=0, stddev=0.01)))
    biase_list.append(tf.Variable(tf.zeros([1])))
    ys_pre = tf.squeeze(tf.matmul(h_list[-1], weight_list[-1]) + biase_list[-1])

    weight_l2 = tf.reduce_sum([tf.nn.l2_loss(w) for w in weight_list])

    target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))
    loss = target_loss + params['beta'] * weight_l2

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        data_size = dtrain.shape[0]
        np.random.shuffle(dtrain)
        np.random.shuffle(dtest)
        store_dict = {}
        store_dict['auc_list'] = []
        store_dict['loss_list'] = []
        store_dict['h_size_list'] = params['h_size_list']
        store_dict['embedding_size'] = params['embedding_size']
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
        store_dict['auc_list'].append(get_auc(t_ys_v, t_ys_pre_v))
        store_dict['loss_list'].append(t_target_loss_v)
        stop_count = 0
        pre_auc = 0.0
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
            store_dict['auc_list'].append(cur_auc)
            store_dict['loss_list'].append(t_target_loss_v)
            if (cur_auc - pre_auc) < 0.0001:
                stop_count += 1
            pre_auc = cur_auc
            if stop_count == 3 or i == (params['round'] - 1):
                break
        if params['rewrite']:
            store_list = -1
        else:
            store_list = load_obj(params['pnn1_params_test_result_file_path'])
        if store_list == -1:
            store_list = [store_dict]
        else:
            store_list.append(store_dict)
        store_obj(store_list, params['pnn1_params_test_result_file_path'])
        print 'pnn1 completed'

# train two pnn, one for link prediction with initialization, one for tow-lop link prediction for assisting, train alternately
def pnn_with_ann_test(params):
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
    p_a = tf.reshape(tf.batch_matmul(tf.expand_dims(e1_a, 2), tf.expand_dims(e2_a, 1)),
                     [-1, params['embedding_size'] * params['embedding_size']])

    h0_a = tf.concat(1, [z_a, p_a])

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
    p_b = tf.reshape(tf.batch_matmul(tf.expand_dims(e1_b, 2), tf.expand_dims(e2_b, 1)),
                     [-1, params['embedding_size'] * params['embedding_size']])

    h0_b = tf.concat(1, [z_b, p_b])

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

    target_loss_b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre_b, ys_b))
    loss_b = target_loss_b + params['beta2'] * weight_l2_b

    train_step_b = tf.train.AdamOptimizer(params['learning_rate2']).minimize(loss_b)

    with tf.Session() as sess:
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
            if (cur_auc - pre_auc) < 0.0001:
                stop_count += 1
            pre_auc = cur_auc
            if stop_count == 4:
                if params['store_test_result']:
                    store_dict = {'u1s': dtest_a[0:800000, 0], 'u2s': dtest_a[0:800000, 1], 'ys': t_ys_a_v, 'ys_pre': t_ys_pre_a_v}
                    store_obj(store_dict, params['pnn2_test_result_file_path'])
                break

# two embedding p and q
def pnn_with_ann_test2(params):
    pre_model = load_obj(params['pre_model_save_path'])
    p = tf.Variable(pre_model['p_v'])
    q = tf.Variable(pre_model['q_v'])
    print(len(pre_model['p_v']))

    dtrain_a = np.loadtxt(params['dtrain_a_file_path'], delimiter='\t')
    dtrain_a -= [1, 1, 0]
    dtest_a = np.loadtxt(params['dtest_a_file_path'], delimiter='\t')
    dtest_a -= [1, 1, 0]

    u1s_a = tf.placeholder(tf.int32, shape=[None])
    u2s_a = tf.placeholder(tf.int32, shape=[None])
    ys_a = tf.placeholder(tf.float32, shape=[None])

    e1_a = tf.nn.embedding_lookup(p, u1s_a)
    e2_a = tf.nn.embedding_lookup(q, u2s_a)

    z_a = tf.concat(1, [e1_a, e2_a])
    p_a = tf.reshape(tf.batch_matmul(tf.expand_dims(e1_a, 2), tf.expand_dims(e2_a, 1)),
                     [-1, params['embedding_size'] * params['embedding_size']])

    h0_a = tf.concat(1, [z_a, p_a])

    weight1_a = tf.Variable(pre_model['weight1_v'])
    biase1_a = tf.Variable(pre_model['biase1_v'])
    h1_a = tf.nn.relu(tf.matmul(h0_a, weight1_a) + biase1_a)

    weight2_a = tf.Variable(pre_model['weight2_v'])
    biase2_a = tf.Variable(pre_model['biase2_v'])
    h2_a = tf.nn.relu(tf.matmul(h1_a, weight2_a) + biase2_a)

    weight3_a = tf.Variable(pre_model['weight3_v'])
    biase3_a = tf.Variable(pre_model['biase3_v'])
    h3_a = tf.nn.relu(tf.matmul(h2_a, weight3_a) + biase3_a)

    weight4_a = tf.Variable(pre_model['weight4_v'])
    biase4_a = tf.Variable(pre_model['biase4_v'])
    h4_a = tf.nn.relu(tf.matmul(h3_a, weight4_a) + biase4_a)

    weighto_a = tf.Variable(pre_model['weighto_v'])
    biaseo_a = tf.Variable(pre_model['biaseo_v'])
    ys_pre_a = tf.squeeze(tf.matmul(h4_a, weighto_a) + biaseo_a)

    weight_l2_a = tf.reduce_sum(
        [tf.nn.l2_loss(weight1_a), tf.nn.l2_loss(weight2_a), tf.nn.l2_loss(weight3_a),
         tf.nn.l2_loss(weight4_a),
         tf.nn.l2_loss(weighto_a)])

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
        p_set.add('%d %d' % (i[0], i[1]))
    while (len(n_set) < n_num):
        a = random.randint(1, params['node_num'])
        b = random.randint(1, params['node_num'])
        line = '%d %d' % (a, b)
        if a != b and line not in p_set and line not in n_set:
            n_set.add(line)
    n_list = [[int(s) for s in line.split(' ')] for line in n_set]
    n_train = np.zeros([len(n_list), 3])
    n_train[:, 0:2] = n_list
    dtrain_b = np.concatenate((p_train, n_train), axis=0)
    dtrain_b -= [1, 1, 0]

    u1s_b = tf.placeholder(tf.int32, shape=[None])
    u2s_b = tf.placeholder(tf.int32, shape=[None])
    ys_b = tf.placeholder(tf.float32, shape=[None])

    e1_b = tf.nn.embedding_lookup(p, u1s_b)
    e2_b = tf.nn.embedding_lookup(q, u2s_b)

    z_b = tf.concat(1, [e1_b, e2_b])
    p_b = tf.reshape(tf.batch_matmul(tf.expand_dims(e1_b, 2), tf.expand_dims(e2_b, 1)),
                     [-1, params['embedding_size'] * params['embedding_size']])

    h0_b = tf.concat(1, [z_b, p_b])

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

    weight4_b = tf.Variable(tf.random_normal([params['h3_size'], params['h4_size']], mean=0, stddev=0.01))
    biase4_b = tf.Variable(tf.zeros([params['h4_size']]))
    h4_b = tf.nn.relu(tf.matmul(h3_b, weight4_b) + biase4_b)

    weighto_b = tf.Variable(tf.random_normal([params['h4_size'], 1], mean=0, stddev=0.01))
    biaseo_b = tf.Variable(tf.zeros([1]))
    ys_pre_b = tf.squeeze(tf.matmul(h4_b, weighto_b) + biaseo_b)

    weight_l2_b = tf.reduce_sum(
        [tf.nn.l2_loss(weight1_b), tf.nn.l2_loss(weight2_b), tf.nn.l2_loss(weight3_b),
         tf.nn.l2_loss(weight4_b),
         tf.nn.l2_loss(weighto_b)])

    target_loss_b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre_b, ys_b))
    loss_b = target_loss_b + params['beta2'] * weight_l2_b

    train_step_b = tf.train.AdamOptimizer(params['learning_rate2']).minimize(loss_b)

    with tf.Session() as sess:
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
            if (cur_auc - pre_auc) < 0.0001:
                stop_count += 1
            pre_auc = cur_auc
            if stop_count == 4:
                if params['store_test_result']:
                    store_dict = {'u1s': dtest_a[0:800000, 0], 'u2s': dtest_a[0:800000, 1], 'ys': t_ys_a_v,
                                  'ys_pre': t_ys_pre_a_v}
                    store_obj(store_dict, params['pnn2_test_result_file_path'])
                break

# without pre-training
def pnn_with_ann_test3(params):
    embedding = tf.Variable(tf.random_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))

    dtrain_a = np.loadtxt(params['dtrain_a_file_path'], delimiter='\t')
    dtrain_a -= [1, 1, 0]
    dtest_a = np.loadtxt(params['dtest_a_file_path'], delimiter='\t')
    dtest_a -= [1, 1, 0]

    u1s_a = tf.placeholder(tf.int32, shape=[None])
    u2s_a = tf.placeholder(tf.int32, shape=[None])
    ys_a = tf.placeholder(tf.float32, shape=[None])

    e1_a = tf.nn.embedding_lookup(embedding, u1s_a)
    e2_a = tf.nn.embedding_lookup(embedding, u2s_a)

    z_a = tf.concat(1, [e1_a, e2_a])
    p_a = tf.reshape(tf.batch_matmul(tf.expand_dims(e1_a, 2), tf.expand_dims(e2_a, 1)),
                     [-1, params['embedding_size'] * params['embedding_size']])

    h0_a = tf.concat(1, [z_a, p_a])

    weight1_a = tf.Variable(pre_model['weight1_v'])
    biase1_a = tf.Variable(pre_model['biase1_v'])
    h1_a = tf.nn.relu(tf.matmul(h0_a, weight1_a) + biase1_a)

    weight2_a = tf.Variable(pre_model['weight2_v'])
    biase2_a = tf.Variable(pre_model['biase2_v'])
    h2_a = tf.nn.relu(tf.matmul(h1_a, weight2_a) + biase2_a)

    weight3_a = tf.Variable(pre_model['weight3_v'])
    biase3_a = tf.Variable(pre_model['biase3_v'])
    h3_a = tf.nn.relu(tf.matmul(h2_a, weight3_a) + biase3_a)

    weight4_a = tf.Variable(pre_model['weight4_v'])
    biase4_a = tf.Variable(pre_model['biase4_v'])
    h4_a = tf.nn.relu(tf.matmul(h3_a, weight4_a) + biase4_a)

    weighto_a = tf.Variable(pre_model['weighto_v'])
    biaseo_a = tf.Variable(pre_model['biaseo_v'])
    ys_pre_a = tf.squeeze(tf.matmul(h4_a, weighto_a) + biaseo_a)

    weight_l2_a = tf.reduce_sum(
        [tf.nn.l2_loss(weight1_a), tf.nn.l2_loss(weight2_a), tf.nn.l2_loss(weight3_a),
         tf.nn.l2_loss(weight4_a),
         tf.nn.l2_loss(weighto_a)])

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
        p_set.add('%d %d' % (i[0], i[1]))

    while (len(n_set) < n_num):
        a = random.randint(1, params['node_num'])
        b = random.randint(1, params['node_num'])
        line = '%d %d' % (a, b)
        if a != b and line not in p_set and line not in n_set:
            n_set.add(line)

    n_list = [[int(s) for s in line.split(' ')] for line in n_set]
    n_train = np.zeros([len(n_list), 3])
    n_train[:, 0:2] = n_list
    dtrain_b = np.concatenate((p_train, n_train), axis=0)
    dtrain_b -= [1, 1, 0]

    u1s_b = tf.placeholder(tf.int32, shape=[None])
    u2s_b = tf.placeholder(tf.int32, shape=[None])
    ys_b = tf.placeholder(tf.float32, shape=[None])

    e1_b = tf.nn.embedding_lookup(embedding, u1s_b)
    e2_b = tf.nn.embedding_lookup(embedding, u2s_b)

    z_b = tf.concat(1, [e1_b, e2_b])
    p_b = tf.reshape(tf.batch_matmul(tf.expand_dims(e1_b, 2), tf.expand_dims(e2_b, 1)),
                     [-1, params['embedding_size'] * params['embedding_size']])

    h0_b = tf.concat(1, [z_b, p_b])

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

    weight4_b = tf.Variable(tf.random_normal([params['h3_size'], params['h4_size']], mean=0, stddev=0.01))
    biase4_b = tf.Variable(tf.zeros([params['h4_size']]))
    h4_b = tf.nn.relu(tf.matmul(h3_b, weight4_b) + biase4_b)

    weighto_b = tf.Variable(tf.random_normal([params['h4_size'], 1], mean=0, stddev=0.01))
    biaseo_b = tf.Variable(tf.zeros([1]))
    ys_pre_b = tf.squeeze(tf.matmul(h4_b, weighto_b) + biaseo_b)

    weight_l2_b = tf.reduce_sum(
        [tf.nn.l2_loss(weight1_b), tf.nn.l2_loss(weight2_b), tf.nn.l2_loss(weight3_b),
         tf.nn.l2_loss(weight4_b),
         tf.nn.l2_loss(weighto_b)])

    target_loss_b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre_b, ys_b))
    loss_b = target_loss_b + params['beta2'] * weight_l2_b

    train_step_b = tf.train.AdamOptimizer(params['learning_rate2']).minimize(loss_b)

    with tf.Session() as sess:
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
            if (cur_auc - pre_auc) < 0.0001:
                stop_count += 1
            pre_auc = cur_auc
            if stop_count == 4:
                if params['store_test_result']:
                    store_dict = {'u1s': dtest_a[0:800000, 0], 'u2s': dtest_a[0:800000, 1], 'ys': t_ys_a_v,
                                  'ys_pre': t_ys_pre_a_v}
                    store_obj(store_dict, params['pnn2_test_result_file_path'])
                break

# pnn with outer product
def pnn(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter='\t')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter='\t')
    dtest -= [1, 1, 0]

    embedding = tf.Variable(
        tf.random_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(embedding, u1s)
    e2 = tf.nn.embedding_lookup(embedding, u2s)

    z = tf.concat(1, [e1, e2])
    p = tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)),
                   [-1, params['embedding_size'] * params['embedding_size']])

    h0 = tf.concat(1, [z, p])

    weight1 = tf.Variable(
        tf.random_normal([params['embedding_size'] * (params['embedding_size'] + 2), params['h1_size']],
                         mean=0,
                         stddev=0.01))
    biase1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.nn.relu(tf.matmul(h0, weight1) + biase1)

    weight2 = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0, stddev=0.01))
    biase2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.nn.relu(tf.matmul(h1, weight2) + biase2)

    weight3 = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0, stddev=0.01))
    biase3 = tf.Variable(tf.zeros([params['h3_size']]))
    h3 = tf.nn.relu(tf.matmul(h2, weight3) + biase3)

    weight4 = tf.Variable(tf.random_normal([params['h3_size'], params['h4_size']], mean=0, stddev=0.01))
    biase4 = tf.Variable(tf.zeros([params['h4_size']]))
    h4 = tf.nn.relu(tf.matmul(h3, weight4) + biase4)

    weighto = tf.Variable(tf.random_normal([params['h4_size'], 1], mean=0, stddev=0.01))
    biaseo = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h4, weighto) + biaseo)

    weight_l2 = tf.reduce_sum(
        [tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3), tf.nn.l2_loss(weight4),
         tf.nn.l2_loss(weighto)])

    target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))
    # target_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(ys_pre, ys, 6))
    loss = target_loss + params['beta'] * weight_l2

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
            if (cur_auc - pre_auc) < 0.0001:
                stop_count += 1
            pre_auc = cur_auc
            if stop_count == 3 or i == (params['round'] - 1):
                weight1_v, biase1_v, weight2_v, biase2_v, weight3_v, biase3_v, weight4_v, biase4_v, weighto_v, biaseo_v, embedding_v = sess.run(
                    [weight1, biase1, weight2, biase2, weight3, biase3, weight4, biase4, weighto, biaseo,
                     embedding],
                    feed_dict={u1s: dtrain[0:0, 0], u2s: dtrain[0:0, 1], ys: np.float32(dtrain[0:0, 2])})
                save_dict = {'weight1_v': weight1_v, 'biase1_v': biase1_v, 'weight2_v': weight2_v,
                             'biase2_v': biase2_v,
                             'weight3_v': weight3_v, 'biase3_v': biase3_v, 'weight4_v': weight4_v,
                             'biase4_v': biase4_v,
                             'weighto_v': weighto_v, 'biaseo_v': biaseo_v, 'embedding_v': embedding_v}
                store_obj(save_dict, params['model_save_path'])
                break

# train two pnn, one for link prediction with initialization, one for tow-lop link prediction for assisting, train alternately
def pnn_with_ann(params):
    pre_model = load_obj(params['pre_model_save_path'])
    embedding = tf.Variable(pre_model['embedding_v'])

    dtrain_a = np.loadtxt(params['dtrain_a_file_path'], delimiter='\t')
    dtrain_a -= [1, 1, 0]
    dtest_a = np.loadtxt(params['dtest_a_file_path'], delimiter='\t')
    dtest_a -= [1, 1, 0]

    u1s_a = tf.placeholder(tf.int32, shape=[None])
    u2s_a = tf.placeholder(tf.int32, shape=[None])
    ys_a = tf.placeholder(tf.float32, shape=[None])

    e1_a = tf.nn.embedding_lookup(embedding, u1s_a)
    e2_a = tf.nn.embedding_lookup(embedding, u2s_a)

    z_a = tf.concat(1, [e1_a, e2_a])
    p_a = tf.reshape(tf.batch_matmul(tf.expand_dims(e1_a, 2), tf.expand_dims(e2_a, 1)),
                     [-1, params['embedding_size'] * params['embedding_size']])

    h0_a = tf.concat(1, [z_a, p_a])

    weight1_a = tf.Variable(pre_model['weight1_v'])
    biase1_a = tf.Variable(pre_model['biase1_v'])
    h1_a = tf.nn.relu(tf.matmul(h0_a, weight1_a) + biase1_a)

    weight2_a = tf.Variable(pre_model['weight2_v'])
    biase2_a = tf.Variable(pre_model['biase2_v'])
    h2_a = tf.nn.relu(tf.matmul(h1_a, weight2_a) + biase2_a)

    weight3_a = tf.Variable(pre_model['weight3_v'])
    biase3_a = tf.Variable(pre_model['biase3_v'])
    h3_a = tf.nn.relu(tf.matmul(h2_a, weight3_a) + biase3_a)

    weight4_a = tf.Variable(pre_model['weight4_v'])
    biase4_a = tf.Variable(pre_model['biase4_v'])
    h4_a = tf.nn.relu(tf.matmul(h3_a, weight4_a) + biase4_a)

    weighto_a = tf.Variable(pre_model['weighto_v'])
    biaseo_a = tf.Variable(pre_model['biaseo_v'])
    ys_pre_a = tf.squeeze(tf.matmul(h4_a, weighto_a) + biaseo_a)

    weight_l2_a = tf.reduce_sum(
        [tf.nn.l2_loss(weight1_a), tf.nn.l2_loss(weight2_a), tf.nn.l2_loss(weight3_a), tf.nn.l2_loss(weight4_a),
         tf.nn.l2_loss(weighto_a)])

    target_loss_a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre_a, ys_a))
    loss_a = target_loss_a + params['beta'] * weight_l2_a

    train_step_a = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss_a)

    dtrain_b = np.loadtxt(params['dtrain_b_file_path'], delimiter='\t')
    dtrain_b -= [1, 1, 0]

    u1s_b = tf.placeholder(tf.int32, shape=[None])
    u2s_b = tf.placeholder(tf.int32, shape=[None])
    ys_b = tf.placeholder(tf.float32, shape=[None])

    e1_b = tf.nn.embedding_lookup(embedding, u1s_b)
    e2_b = tf.nn.embedding_lookup(embedding, u2s_b)

    z_b = tf.concat(1, [e1_b, e2_b])
    p_b = tf.reshape(tf.batch_matmul(tf.expand_dims(e1_b, 2), tf.expand_dims(e2_b, 1)),
                     [-1, params['embedding_size'] * params['embedding_size']])

    h0_b = tf.concat(1, [z_b, p_b])

    weight1_b = tf.Variable(
        tf.random_normal([params['embedding_size'] * (params['embedding_size'] + 2), params['h1_size']], mean=0,
                         stddev=0.01))
    biase1_b = tf.Variable(tf.zeros([params['h1_size']]))
    h1_b = tf.nn.relu(tf.matmul(h0_b, weight1_b) + biase1_b)

    weight2_b = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0, stddev=0.01))
    biase2_b = tf.Variable(tf.zeros([params['h2_size']]))
    h2_b = tf.nn.relu(tf.matmul(h1_b, weight2_b) + biase2_b)

    weight3_b = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0, stddev=0.01))
    biase3_b = tf.Variable(tf.zeros([params['h3_size']]))
    h3_b = tf.nn.relu(tf.matmul(h2_b, weight3_b) + biase3_b)

    weight4_b = tf.Variable(tf.random_normal([params['h3_size'], params['h4_size']], mean=0, stddev=0.01))
    biase4_b = tf.Variable(tf.zeros([params['h4_size']]))
    h4_b = tf.nn.relu(tf.matmul(h3_b, weight4_b) + biase4_b)

    weighto_b = tf.Variable(tf.random_normal([params['h4_size'], 1], mean=0, stddev=0.01))
    biaseo_b = tf.Variable(tf.zeros([1]))
    ys_pre_b = tf.squeeze(tf.matmul(h4_b, weighto_b) + biaseo_b)

    weight_l2_b = tf.reduce_sum(
        [tf.nn.l2_loss(weight1_b), tf.nn.l2_loss(weight2_b), tf.nn.l2_loss(weight3_b), tf.nn.l2_loss(weight4_b),
         tf.nn.l2_loss(weighto_b)])

    target_loss_b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre_b, ys_b))
    loss_b = target_loss_b + params['beta'] * weight_l2_b

    train_step_b = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss_b)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print 'training pnn2...'

        data_size_a = dtrain_a.shape[0]
        np.random.shuffle(dtrain_a)
        np.random.shuffle(dtest_a)
        loss_a_v, target_loss_a_v, ys_a_v, ys_pre_a_v = sess.run([loss_a, target_loss_a, ys_a, tf.sigmoid(ys_pre_a)],
                                                                 feed_dict={u1s_a: dtrain_a[0:800000, 0],
                                                                            u2s_a: dtrain_a[0:800000, 1],
                                                                            ys_a: np.float32(dtrain_a[0:800000, 2])})
        t_loss_a_v, t_target_loss_a_v, t_ys_a_v, t_ys_pre_a_v = sess.run(
            [loss_a, target_loss_a, ys_a, tf.sigmoid(ys_pre_a)],
            feed_dict={u1s_a: dtest_a[0:800000, 0], u2s_a: dtest_a[0:800000, 1],
                       ys_a: np.float32(dtest_a[0:800000, 2])})

        data_size_b = dtrain_b.shape[0]
        np.random.shuffle(dtrain_b)
        loss_b_v, target_loss_b_v, ys_b_v, ys_pre_b_v = sess.run([loss_b, target_loss_b, ys_b, tf.sigmoid(ys_pre_b)],
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
                i + 1, loss_a_v, target_loss_a_v, get_auc(ys_a_v, ys_pre_a_v), t_loss_a_v, t_target_loss_a_v,
                get_auc(t_ys_a_v, t_ys_pre_a_v), loss_b_v, target_loss_b_v, get_auc(ys_b_v, ys_pre_b_v))

            cur_auc = get_auc(t_ys_a_v, t_ys_pre_a_v)
            if (cur_auc - pre_auc) < 0.0001:
                stop_count += 1
            pre_auc = cur_auc
            if stop_count == 4:
                break

def show_pnn1_result(params):
    # pre_model = load_obj(params['pre_model_save_path'])
    # embedding = tf.Variable(pre_model['embedding_v'])

    dtest_a = np.loadtxt(params['dtest_a_file_path'], delimiter='\t')
    dtest_a -= [1, 1, 0]

    # u1s_a = tf.placeholder(tf.int32, shape=[None])
    # u2s_a = tf.placeholder(tf.int32, shape=[None])
    # ys_a = tf.placeholder(tf.float32, shape=[None])
    #
    # e1_a = tf.nn.embedding_lookup(embedding, u1s_a)
    # e2_a = tf.nn.embedding_lookup(embedding, u2s_a)
    #
    # z_a = tf.concat(1, [e1_a, e2_a])eu
    # p_a = tf.reshape(tf.batch_matmul(tf.expand_dims(e1_a, 2), tf.expand_dims(e2_a, 1)),
    #                  [-1, params['embedding_size'] * params['embedding_size']])
    #
    # h0_a = tf.concat(1, [z_a, p_a])
    #
    # weight1_a = tf.Variable(pre_model['weight1_v'])
    # biase1_a = tf.Variable(pre_model['biase1_v'])
    # h1_a = tf.nn.relu(tf.matmul(h0_a, weight1_a) + biase1_a)
    #
    # weight2_a = tf.Variable(pre_model['weight2_v'])
    # biase2_a = tf.Variable(pre_model['biase2_v'])
    # h2_a = tf.nn.relu(tf.matmul(h1_a, weight2_a) + biase2_a)
    #
    # weight3_a = tf.Variable(pre_model['weight3_v'])
    # biase3_a = tf.Variable(pre_model['biase3_v'])
    # h3_a = tf.nn.relu(tf.matmul(h2_a, weight3_a) + biase3_a)
    #
    # weight4_a = tf.Variable(pre_model['weight4_v'])
    # biase4_a = tf.Variable(pre_model['biase4_v'])
    # h4_a = tf.nn.relu(tf.matmul(h3_a, weight4_a) + biase4_a)
    #
    # weighto_a = tf.Variable(pre_model['weighto_v'])
    # biaseo_a = tf.Variable(pre_model['biaseo_v'])
    # ys_pre_a = tf.squeeze(tf.matmul(h4_a, weighto_a) + biaseo_a)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())
    #     np.random.shuffle(dtest_a)
    #     ys_a_t, ys_pre_a_t = sess.run([ys_a, tf.sigmoid(ys_pre_a)], feed_dict={u1s_a: dtest_a[:, 0], u2s_a: dtest_a[:, 1], ys_a: dtest_a[:, 2]})
    #     count = 0
    #     i = 0
    #     while(count<1000):
    #         if ys_a_t[i]>0.5:
    #             print '%2f\t%2f' % (ys_a_t[i], ys_pre_a_t[i])
    #             count+=1
    #         i+=1
    #
    #     count = 0
    #     i = 0
    #     while (count < 1000):
    #         if ys_a_t[i]<0.5:
    #             print '%2f\t%2f' % (ys_a_t[i], ys_pre_a_t[i])
    #             count += 1
    #         i += 1

def show_params_test_result(params):
    path_list = params['source_file_path'].split('/')
    dir = '/'.join(path_list[0:-1]) + '/'
    data_name = path_list[-1].split('_')[0]
    version = params['version']
    path = dir + '%s_pnn1_params_test_result_v%d' % (data_name, version)

    store_list = load_obj(path)
    if store_list == -1:
        print('gg, my friend')
        return
    color_list = ['r', 'g', 'b']
    segment_list = [[0, 6]]#
    # for store_dict in store_list:
    #     print(store_dict['embedding_size'])
    #     print(store_dict['h_size_list'])
    #     print(store_dict['auc_list'])
    #     print(store_dict['loss_list'])
    #     print('')
    print len(store_list)
    for i in range(len(segment_list)):
        l = segment_list[i]
        x_list = range(1, l[1]-l[0]+1)
        y_list = [max(t['auc_list']) for t in store_list[l[0]:l[1]]]
        print x_list,y_list
        pl.plot(x_list, y_list, color_list[i])
    pl.ylim(0.97,0.99)
    pl.show()

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
    # return roc_auc_score(y_trues, y_pres)
    fpr, tpr, thresholds = metrics.roc_curve(np.int32(y_trues), y_pres, pos_label=1)
    # print('auc1: %f, auc2: %f'%(roc_auc_score(y_trues, y_pres), metrics.auc(fpr, tpr)))
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
                         'store_test_result': params['store_test_result'],
                         'mf_test_result_file_path': dir + '%s_mf_test_result_v%d' % (data_name, version)})
    if len(params['pnn1_test'])>0:
        p = params['pnn1_test']
        pnn_test({'dtrain_file_path': dir + '%s_train_data_v%d' % (data_name, version),
             'dtest_file_path': dir + '%s_test_data_v%d' % (data_name, version),
             'model_save_path': dir + '%s_data_pnn_model_params_saver_v%d' % (data_name, version),
             'embedding_size': 20, 'node_num': node_num,
             'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
             'batch_size': 5000, 'round': p['round'], 'learning_rate': 4e-3, 'beta': 4e-4,
             'store_test_result': params['store_test_result'],
             'pnn1_test_result_file_path': dir + '%s_pnn1_test_result_v%d' % (data_name, version)})
    if params['pnn1']:
        pnn({'dtrain_file_path': dir + '%s_train_data_v%d' % (data_name, version),
             'dtest_file_path': dir + '%s_test_data_v%d' % (data_name, version),
             'model_save_path': dir + '%s_data_pnn_model_params_saver_v%d' % (data_name, version),
             'embedding_size': 20, 'node_num': node_num,
             'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
             'batch_size': 5000, 'round': 30, 'learning_rate': 4e-3, 'beta': 4e-4})
    if params['fixed_emb_pnn2']:
        fixed_emb_pnn2({'dtrain_file_path': dir + '%s_hop2_train_data_v%d' % (data_name, version),
                      'pre_model_save_path': dir + '%s_data_pnn_model_params_saver_v%d' % (data_name, version),
                      'embedding_size': 20, 'node_num': node_num,
                      'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
                      'batch_size': 5000, 'round': 15, 'learning_rate': 4e-3, 'beta': 4e-4})
    if len(params['pnn2_test'])>0:
        p = params['pnn2_test']
        pnn_with_ann_test({'dtrain_a_file_path': dir + '%s_train_data_v%d' % (data_name, version),
                           'dtest_a_file_path': dir + '%s_test_data_v%d' % (data_name, version),
                           'dtrain_b_file_path': dir + '%s_hop2_train_positive_data_v%d' % (data_name, version),
                           'pre_model_save_path': dir + '%s_data_pnn_model_params_saver_v%d' % (data_name, version),
                           'embedding_size': 20, 'node_num': node_num,
                           'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
                           'batch_size': 5000, 'round': p['round'], 'learning_rate1': p['learning_rate1'], 'learning_rate2': p['learning_rate2'], 'beta1': p['beta1'], 'beta2': p['beta2'],
                           'hop2_np_rate': p['hop2_np_rate'],
                           'store_test_result': params['store_test_result'],
                           'pnn2_test_result_file_path': dir + '%s_pnn2_test_result_v%d' % (data_name, version)
                           })
    if params['pnn2']:
        pnn_with_ann({'dtrain_a_file_path': dir + '%s_train_data_v%d' % (data_name, version),
                      'dtest_a_file_path': dir + '%s_test_data_v%d' % (data_name, version),
                      'dtrain_b_file_path': dir + '%s_hop2_train_data_v%d' % (data_name, version),
                      'pre_model_save_path': dir + '%s_data_pnn_model_params_saver_v%d' % (data_name, version),
                      'embedding_size': 20, 'node_num': node_num,
                      'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
                      'batch_size': 5000, 'round': 15, 'learning_rate': 4e-3, 'beta': 4e-4})
    if params['show_auc_curve']:
        show_auc_curve({'mf_test_result_file_path': dir + '%s_mf_test_result_v%d' % (data_name, version),
                        'pnn1_test_result_file_path': dir + '%s_pnn1_test_result_v%d' % (data_name, version),
                        'pnn2_test_result_file_path': dir + '%s_pnn2_test_result_v%d' % (data_name, version),
                        'neighbor_set_list_file_path': dir + '%s_train_neighbor_set_list_v%d' % (data_name, version),
                        'x_max': 10,
                        'x_step': 1})
    # if len(params['pnn1_test2'])>0:
    #     p = params['pnn1_test2']
    #     pnn_test2({'dtrain_file_path': dir + '%s_train_data_v%d' % (data_name, version),
    #          'dtest_file_path': dir + '%s_test_data_v%d' % (data_name, version),
    #          'model_save_path': dir + '%s_data_pnn_model_params_saver_tmp_v%d' % (data_name, version),
    #          'embedding_size': 20, 'node_num': node_num,
    #          'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #          'batch_size': 5000, 'round': p['round'], 'learning_rate': 4e-3, 'beta': 4e-4,
    #          'store_test_result': params['store_test_result'],
    #          'pnn1_test_result_file_path': dir + '%s_pnn1_test_result_tmp_v%d' % (data_name, version)})
    # if len(params['pnn2_test2'])>0:
    #     p = params['pnn2_test2']
    #     pnn_with_ann_test2({'dtrain_a_file_path': dir + '%s_train_data_v%d' % (data_name, version),
    #                        'dtest_a_file_path': dir + '%s_test_data_v%d' % (data_name, version),
    #                        'dtrain_b_file_path': dir + '%s_hop2_train_positive_data_v%d' % (data_name, version),
    #                        'pre_model_save_path': dir + '%s_data_pnn_model_params_saver_tmp_v%d' % (data_name, version),
    #                        'embedding_size': 20, 'node_num': node_num,
    #                        'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #                        'batch_size': 5000, 'round': p['round'], 'learning_rate1': p['learning_rate1'], 'learning_rate2': p['learning_rate2'], 'beta1': p['beta1'], 'beta2': p['beta2'],
    #                        'hop2_np_rate': p['hop2_np_rate'],
    #                        'store_test_result': params['store_test_result'],
    #                        'pnn2_test_result_file_path': dir + '%s_pnn2_test_result_tmp_v%d' % (data_name, version)
    #                        })

def ov_pnn1(params):
    path_list = params['source_file_path'].split('/')
    dir = '/'.join(path_list[0:-1]) + '/'
    data_name = path_list[-1].split('_')[0]
    version = params['version']

    pnn_test3({'dtrain_file_path': dir + '%s_train_data_v%d' % (data_name, version),
               'dtest_file_path': dir + '%s_test_data_v%d' % (data_name, version),
               'embedding_size': params['embedding_size'], 'node_num': params['node_num'],
               'h_size_list': params['h_size_list'],
               'batch_size': 5000, 'round': 25, 'learning_rate': 4e-3, 'beta': 4e-4,
               'pnn1_params_test_result_file_path': dir + '%s_pnn1_params_test_result_v%d' % (data_name, version),
               'rewrite': params['rewrite']})

def params_exp(params):
    node_num = get_max_node_num(params['source_file_path'])
    if len(params['embedding_size_list'])==1:
        for i in range(len(params['h_size_list'])):
            if i == 1:
                params['rewrite'] = False
            h_list = params['h_size_list'][i]
            ov_pnn1({'source_file_path': params['source_file_path'],
                'version': 1,
                'embedding_size': params['embedding_size_list'][0],
                'h_size_list': h_list,
                'node_num': node_num,
                'rewrite': params['rewrite']
                })
    else:
        for i in range(len(params['embedding_size_list'])):
            if i == 1:
                params['rewrite'] = False
            e_list = params['embedding_size_list'][i]
            ov_pnn1({'source_file_path': params['source_file_path'],
                'version': 1,
                'embedding_size': e_list,
                'h_size_list': params['h_size_list'][0],
                'node_num': node_num,
                'rewrite': params['rewrite']
                })

def np_rate_exp(params):
    print('coding')

def pnn2_without_pt_exp(params):
    print('coding')

def completeness_exp(params):
    print('coding')

if __name__ == '__main__':
    # base_exp({'source_file_path': '../../data/test1/cora_data',
    #           'version': 1,
    #           'train_np_rate': 160,
    #           'baseline_set': set(['cn', 'aa', 'ra', 'katz', 'rwr', 'mf']),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4,
    #                         'hop2_np_rate': 160, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': True})

    # params_exp({'source_file_path': '../../data/test1/openflights_data',
    #             'version': 1,
    #             'embedding_size_list': [20],
    #             'h_size_list': [[20], [20, 20], [20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20, 20]],
    #             'rewrite': True
    #             })

    # show_params_test_result({'source_file_path': '../../data/test1/openflights_data', 'version': 1})

    base_exp({'source_file_path': '../../data/test1/openflights_data',
              'version': 1,
              'train_np_rate': 160,
              'baseline_set': set(['cn', 'aa', 'ra', 'katz', 'rwr', 'mf']),
              'pnn1_test': {'round': 25},
              'pnn1': False,
              'fixed_emb_pnn2': False,
              'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4, 'hop2_np_rate': 40, 'round': 25},
              'pnn2': False,
              'store_test_result': True,
              'show_auc_curve': True})