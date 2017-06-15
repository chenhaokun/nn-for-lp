from sklearn.metrics import roc_auc_score
from scipy.sparse.linalg import spsolve
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from divide_data_e2e import *
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

def show_auc_curve(params):
    # fig = plt.figure()
    #
    # x = np.arange(20)
    # y1 = np.cos(x)
    # y2 = (x ** 2)
    # y3 = (x ** 3)
    # yn = (y1, y2, y3)
    # COLORS = ('b', 'g', 'k')
    #
    # for i, y in enumerate(yn):
    #     ax = fig.add_subplot(len(yn), 1, i + 1)
    #
    #     ax.plot(x, y, ls='solid', color=COLORS[i])
    #
    #     if i != len(yn) - 1:
    #         # all but last
    #         ax.set_xticklabels(())
    #     else:
    #         for tick in ax.xaxis.get_major_ticks():
    #             tick.label.set_fontsize(14)
    #             # specify integer or one of preset strings, e.g.
    #             # tick.label.set_fontsize('x-small')
    #             tick.label.set_rotation('vertical')
    #
    # fig.suptitle('Matplotlib xticklabels Example')
    # plt.show()

    # fig = plt.figure()  # Creates a new figure
    # fig.suptitle('Temperature', fontsize=50)  # Add the text/suptitle to figure
    #
    # ax = fig.add_subplot(111)  # add a subplot to the new figure, 111 means "1x1 grid, first subplot"
    # fig.subplots_adjust(top=0.80)  # adjust the placing of subplot, adjust top, bottom, left and right spacing
    # ax.set_title('Humidity', fontsize=30)  # title of plot
    #
    # ax.set_xlabel('xlabel', fontsize=20)  # xlabel
    # ax.set_ylabel('ylabel', fontsize=20)  # ylabel
    #
    # x = [0, 1, 2, 5, 6, 7, 4, 4, 7, 8]
    # y = [2, 4, 6, 4, 6, 7, 5, 4, 5, 7]
    #
    # ax.plot(x, y, '-o')  # plotting the data with marker '-o'
    # ax.axis([0, 10, 0, 10])  # specifying plot axes lengths
    # plt.show()

    neighbor_set_list = load_obj(params['neighbor_set_list_file_path'])
    points = (params['x_max'] - params['x_min'])/params['x_step']
    x_list = range(params['x_min']+params['x_step']-1, params['x_max'], params['x_step'])
    m_list = ['pnn2', 'pnn1', 'mf']#['pnn2', 'pnn1', 'mf']
    plot_list = []
    color_list = ['b', 'g', 'r']

    for j in range(len(m_list)):
        step_ys_list = [[] for i in range(points)]
        step_ys_pre_list = [[] for i in range(points)]
        result_dict = load_obj(params[m_list[j]+'_test_result_file_path'])
        # print params[m_list[j]+'_test_result_file_path']
        # print result_dict
        u1s_list = result_dict['u1s']
        u2s_list = result_dict['u2s']
        ys_list = result_dict['ys']
        ys_pre_list = result_dict['ys_pre']
        for i in range(len(u1s_list)):
            i1 = len(neighbor_set_list[int(u1s_list[i])][0])
            o1 = len(neighbor_set_list[int(u1s_list[i])][1])
            i2 = len(neighbor_set_list[int(u2s_list[i])][0])
            o2 = len(neighbor_set_list[int(u2s_list[i])][1])
            neighbor_num = min(i1+o1, i2+o2)
            if neighbor_num >= params['x_min'] and neighbor_num < params['x_max']:
                index = (neighbor_num-params['x_min'])/params['x_step']
                step_ys_list[index].append(ys_list[i])
                step_ys_pre_list[index].append(ys_pre_list[i])
        auc_list = []
        tmp_list1 = []
        tmp_list2 = []
        # if j == 2:
        #     count = 0
        #     for i in range(len(step_ys_list[0])):
        #         if step_ys_list[0][i]==0.0 and step_ys_pre_list[0][i]>0.01:
        #             count += 1
        #             print('%f\t%f'%(step_ys_list[0][i], step_ys_pre_list[0][i]))
        #     print count
        for i in range(points):
            tmp_list1 = np.concatenate((step_ys_list[i], tmp_list1))
            tmp_list2 = np.concatenate((step_ys_pre_list[i], tmp_list2))
            # print '%d,%d,%.4f'%(np.sum(tmp_list1),len(tmp_list1),np.sum(tmp_list1)/1./len(tmp_list1))

            # tmp_list1=np.array(step_ys_list[i])
            # tmp_list2 = np.array(step_ys_pre_list[i])
            # if i==0 and j==2:
            #     print tmp_list1, tmp_list2
            # print tmp_list1
            auc_list.append(get_auc(tmp_list1, tmp_list2))
            # print sum(tmp_list1)
        # print(m_list[j])
        # for i in range(points):
        #     print(auc_list[i])
        # if j==1:
        #     auc_list=[auc_list[i]-0.002 for i in range(len(auc_list))]
        tmpplot, = pl.plot(x_list, auc_list, color_list[j], linewidth=2.0)
        plot_list.append(tmpplot)

    pl.title('cold start auc curve in %s'%'pokec', fontsize=24)  # give plot a title
    pl.xlabel('max node degree', fontsize=24)  # make axis labels
    pl.ylabel('auc', fontsize=24)

    # pl.xlim(1, 10)  # set axis limits
    # pl.ylim(params['y_min'], params['y_max'])
    pl.tick_params(labelsize=20)

    pl.legend(tuple(plot_list), ('pnn2', 'pnn1', 'mf'), fontsize=24)  # make legend
    pl.show()  # show the plot on the screen

def show_auc_curve_by_user(params):
    neighbor_set_list = load_obj(params['neighbor_set_list_file_path'])
    points = (params['x_max'] - params['x_min'])/params['x_step']
    x_list = range(params['x_min']+params['x_step']-1, params['x_max'], params['x_step'])
    m_list = ['pnn2', 'pnn1', 'mf']#['pnn2', 'pnn1', 'mf']
    plot_list = []
    color_list = ['b', 'g', 'r']

    alist=[]
    for j in range(len(m_list)):
        result_dict = load_obj(params[m_list[j]+'_test_result_file_path'])
        print get_auc(result_dict['ys'],result_dict['ys_pre'])
        x={}
        d={}
        for u1,u2,ys,pre in zip(result_dict['u1s'],result_dict['u2s'],result_dict['ys'],result_dict['ys_pre']):
            u1=int(u1)
            u2=int(u2)
            if not u1 in x:
                x[u1]=[]
                d[u1]=len(neighbor_set_list[u1][0])+len(neighbor_set_list[u1][1])
            x[u1].append((ys,pre))
        auc_list = []

        aucs = []
        for l in x_list:

            for u in x:
                if d[u]>=l and d[u]<l+params['x_step']:
                    truth=np.array(map(lambda x:x[0],x[u]))
                    pred=np.array(map(lambda x:x[1],x[u]))
                    if np.sum(truth)>0:
                        aucs.append(get_auc(truth, pred))
            auc=sum(aucs)/len(aucs)
            print '%d - %.4f (%d)'%(l,auc,len(aucs))
            auc_list.append(auc)

        tmpplot, = pl.plot(x_list, auc_list, color_list[j], linewidth=2.0)
        plot_list.append(tmpplot)
        alist.append(auc_list)

    pl.title('cold start auc curve in %s'%'pokec', fontsize=24)  # give plot a title
    pl.xlabel('max node degree', fontsize=24)  # make axis labels
    pl.ylabel('auc', fontsize=24)

    # pl.xlim(1, 10)  # set axis limits
    # pl.ylim(params['y_min'], params['y_max'])
    pl.tick_params(labelsize=20)

    pl.legend(tuple(plot_list), ('pnn2', 'pnn1', 'mf'), fontsize=24)  # make legend
    pl.plot(map(lambda x:alist[0][x]/alist[1][x],range(len(alist[0]))))
    pl.show()  # show the plot on the screen

def show_auc_curve_by_user_v2(params):
    neighbor_set_list = load_obj(params['neighbor_set_list_file_path'])
    x_list = range(params['x_min']+params['x_step']-1, params['x_max'], params['x_step'])
    m_list = ['pnn2', 'pnn1', 'mf']
    plot_list = []
    color_list = ['b-D', 'g-o', 'r-^']

    # pl.figure()
    # pl.style.use('classic')

    alist=[]
    for j in range(len(m_list)):
        result_dict = load_obj(params[m_list[j]+'_test_result_file_path'])
        print get_auc(result_dict['ys'],result_dict['ys_pre'])
        x={}
        d={}
        for u1,u2,ys,pre in zip(result_dict['u1s'],result_dict['u2s'],result_dict['ys'],result_dict['ys_pre']):
            u1=int(u1)
            u2=int(u2)
            if not u1 in x:
                x[u1]=[]
                d[u1]=len(neighbor_set_list[u1][0])+len(neighbor_set_list[u1][1])
            x[u1].append((ys,pre))
        auc_list = []

        aucs = []
        for l in x_list:

            for u in x:
                if d[u]<=l and d[u]>(l-params['x_step']):#d[u]>=l and d[u]<l+params['x_step']:
                    truth=np.array(map(lambda x:x[0],x[u]))
                    pred=np.array(map(lambda x:x[1],x[u]))
                    if np.sum(truth)>0:
                        aucs.append(get_auc(truth, pred))
            auc=sum(aucs)/len(aucs)
            print '%d - %.4f (%d)'%(l,auc,len(aucs))
            auc_list.append(auc)

        tmpplot, = pl.plot(x_list, auc_list, color_list[j], linewidth=2.0, markersize=10)
        plot_list.append(tmpplot)
        alist.append(auc_list)

    pl.title('Openflights Dataset', fontsize=24)  # give plot a title
    pl.xlabel('Max Node Degree', fontsize=22)  # make axis labels
    pl.ylabel('AUC', fontsize=22)

    pl.xlim(2, 22)  # set axis limits
    # pl.ylim(params['y_min'], params['y_max'])
    pl.xticks(np.arange(5, 21, 5), fontsize=18)
    pl.yticks(np.arange(0.80, 0.99, 0.05), fontsize=18)
    pl.tick_params(labelsize=18)

    pl.legend(tuple(plot_list), ('DLP-A', 'DLP', 'MF'), fontsize=18, ncol=1, loc=4)  # make legend
    pl.grid()
    pl.tight_layout()

    # figure=pl.figure()
    # pl.style.use('classic')
    # axes = figure.add_subplot(111)
    #
    # ptmp, = pl.plot(x_list, map(lambda x:(alist[0][x]-alist[1][x])*100,range(len(alist[0]))), 'r-o', linewidth=2.0, markersize=10)
    # pl.title('Pokec Dataset', fontsize=24)  # give plot a title
    # pl.xlabel('Max Node Degree', fontsize=22)  # make axis labels
    # pl.ylabel('Improvement (AUC)', fontsize=22)
    # pl.xlim(2,22)
    # pl.xticks(np.arange(5,21,5))
    # pl.yticks(np.arange(1.0,5.1,1.0))
    # pl.tick_params(labelsize=18)
    #
    # fmt = '%.2f%%'
    # yticks = mtick.FormatStrFormatter(fmt)
    # axes.yaxis.set_major_formatter(yticks)
    # pl.legend(tuple([ptmp]), ('Improvement (AUC)\n(DLP-A ~ DLP)',''), fontsize=18, ncol=1, loc=1)
    # # plt.xticks(fontsize=20)
    # # plt.yticks(fontsize=20)
    # pl.grid()
    # pl.tight_layout()

    pl.show()  # show the plot on the screen

def show_embedding_distribution(params):
    thred1 = 0.9575#5009
    thred2 = 0.238#5009
    thred3 = 0.274#5009
    plot_list = []

    # buckets_num = 50
    # step_len = 2.0/buckets_num
    # x_list = np.arange(-1, 1, step_len)
    # y_list_tmp = [0 for i in range(int(buckets_num))]

    buckets_num = 50
    max = 8.0
    step_len = max / buckets_num
    x_list = np.arange(0, max, step_len)
    y_list_tmp = [0 for i in range(int(buckets_num))]

    # buckets_num = 50
    # max = 4.0
    # min = -2.0
    # step_len = (max-min) / buckets_num
    # x_list = np.arange(min, max, step_len)
    # y_list_tmp = [0 for i in range(buckets_num)]

    mf_result_dict = load_obj(params['mf_test_result_file_path'])
    p = mf_result_dict['p']
    q = mf_result_dict['q']
    y_list = list(y_list_tmp)
    for u1,u2,ys,ys_pre in zip(mf_result_dict['u1s'], mf_result_dict['u2s'], mf_result_dict['ys'], mf_result_dict['ys_pre']):
        if ys_pre >= thred1:#int(ys) == 1:
            e1 = p[int(u1)]
            e2 = q[int(u2)]
            # cos = cosin(e1, e2)
            # index = int(np.floor((cos+1.0)/step_len))
            disv = dis(e1, e2)
            index = int(np.floor(disv / step_len))
            # dot_v = dot(e1, e2)
            # index = int(np.floor(dot_v-min / step_len))
            if index<0 :
                index = 0
            if index >= buckets_num:
                index = buckets_num-1
            y_list[index]+=1
    print y_list
    tmpplot, = pl.plot(x_list, normalize(y_list),'r')
    plot_list.append(tmpplot)

    pnn_result_dict = load_obj(params['pnn1_test_result_file_path'])
    embedding = pnn_result_dict['embedding']
    y_list = list(y_list_tmp)
    for u1, u2, ys, ys_pre in zip(pnn_result_dict['u1s'], pnn_result_dict['u2s'], pnn_result_dict['ys'], pnn_result_dict['ys_pre']):
        if ys_pre >= thred2:#int(ys) == 1:
            e1 = embedding[int(u1)]
            e2 = embedding[int(u2)]
            # cos = cosin(e1, e2)
            # index = int(np.floor((cos + 1.0) / step_len))
            disv = dis(e1, e2)
            index = int(np.floor(disv / step_len))
            # dot_v = dot(e1, e2)
            # index = int(np.floor(dot_v-min / step_len))
            if index<0 :
                index = 0
            if index >= buckets_num:
                index = buckets_num-1
            y_list[index] += 1
    print y_list
    tmpplot, = pl.plot(x_list, normalize(y_list),'g')
    plot_list.append(tmpplot)

    pnn_result_dict = load_obj(params['pnn2_test_result_file_path'])
    embedding = pnn_result_dict['embedding'][0]
    y_list = list(y_list_tmp)
    for u1, u2, ys, ys_pre in zip(pnn_result_dict['u1s'], pnn_result_dict['u2s'], pnn_result_dict['ys'], pnn_result_dict['ys_pre']):
        if ys_pre >= thred3:#int(ys) == 1:
            e1 = embedding[int(u1)]
            e2 = embedding[int(u2)]
            # cos = cosin(e1, e2)
            # index = int(np.floor((cos + 1.0) / step_len))
            disv = dis(e1, e2)
            index = int(np.floor(disv / step_len))
            # dot_v = dot(e1, e2)
            # index = int(np.floor(dot_v-min / step_len))
            if index<0 :
                index = 0
            if index >= buckets_num:
                index = buckets_num-1
            y_list[index] += 1
    print y_list
    tmpplot, = pl.plot(x_list, normalize(y_list), 'b')
    plot_list.append(tmpplot)

    # buckets_num = 500
    # max = 9.0
    # min = -4.0
    # thred1 = 0.9575  # 5009
    # thred2 = 0.238  # 5009
    # thred3 = 0.274  # 5009
    # step_len = (max - min) / buckets_num
    # x_list = np.arange(min, max, step_len)
    # y_list_tmp = [0 for i in range(int(buckets_num))]

    mf_result_dict = load_obj(params['mf_test_result_file_path'])
    p = mf_result_dict['p']
    q = mf_result_dict['q']
    y_list = list(y_list_tmp)
    for u1, u2, ys, ys_pre in zip(mf_result_dict['u1s'], mf_result_dict['u2s'], mf_result_dict['ys'], mf_result_dict['ys_pre']):
        if ys_pre < thred1:  # int(ys) == 1:
            e1 = p[int(u1)]
            e2 = q[int(u2)]
            # cos = cosin(e1, e2)
            # index = int(np.floor((cos+1.0)/step_len))
            disv = dis(e1, e2)
            index = int(np.floor(disv / step_len))
            # dot_v = dot(e1, e2)
            # index = int(np.floor(dot_v-min / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
    print y_list
    tmpplot, = pl.plot(x_list, normalize(y_list), 'r--')
    plot_list.append(tmpplot)

    pnn_result_dict = load_obj(params['pnn1_test_result_file_path'])
    embedding = pnn_result_dict['embedding']
    y_list = list(y_list_tmp)
    for u1, u2, ys, ys_pre in zip(pnn_result_dict['u1s'], pnn_result_dict['u2s'], pnn_result_dict['ys'],
                                  pnn_result_dict['ys_pre']):
        if ys_pre < thred2:  # int(ys) == 1:
            e1 = embedding[int(u1)]
            e2 = embedding[int(u2)]
            # cos = cosin(e1, e2)
            # index = int(np.floor((cos + 1.0) / step_len))
            disv = dis(e1, e2)
            index = int(np.floor(disv / step_len))
            # dot_v = dot(e1, e2)
            # index = int(np.floor(dot_v-min / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
    print y_list
    tmpplot, = pl.plot(x_list, normalize(y_list), 'g--')
    plot_list.append(tmpplot)

    pnn_result_dict = load_obj(params['pnn2_test_result_file_path'])
    embedding = pnn_result_dict['embedding'][0]
    y_list = list(y_list_tmp)
    for u1, u2, ys, ys_pre in zip(pnn_result_dict['u1s'], pnn_result_dict['u2s'], pnn_result_dict['ys'],
                                  pnn_result_dict['ys_pre']):
        if ys_pre < thred3:  # int(ys) == 1:
            e1 = embedding[int(u1)]
            e2 = embedding[int(u2)]
            # cos = cosin(e1, e2)
            # index = int(np.floor((cos + 1.0) / step_len))
            disv = dis(e1, e2)
            index = int(np.floor(disv / step_len))
            # dot_v = dot(e1, e2)
            # index = int(np.floor(dot_v-min / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
    print y_list
    tmpplot, = pl.plot(x_list, normalize(y_list), 'b--')
    plot_list.append(tmpplot)

    pl.title('euclid distance distribution', fontsize=24)  # give plot a title
    pl.xlabel('euclid distance', fontsize=24)  # make axis labels
    pl.ylabel('probability', fontsize=24)
    # pl.ylim(0,0.18)

    pl.legend(tuple(plot_list), ('MF~P', 'DLPM~P', 'DLPM-N~P', 'MF~N', 'DLPM~N', 'DLPM-N~N'), fontsize=15, ncol=1, loc=1)  # make legend

    pl.show()

def show_embedding_distribution_v2(params):
    type=1#0:cosine,1:euclid,2:inner product
    show_list = [0,2,3,4,6,7]#[2,3,6,7]#[0,1,4,5]

    buckets_num = 0
    max = 0.0
    min = 0.0
    step_len = 0
    x_list = []
    y_list_tmp = []

    if type==0:
        buckets_num = 50
        max = 2.2
        step_len = max / buckets_num
        x_list = np.arange(0, max, step_len)
        y_list_tmp = [0 for i in range(int(buckets_num))]

    if type==1:
        buckets_num = 50
        step_len = 2.0/buckets_num
        x_list = np.arange(-1, 1, step_len)
        y_list_tmp = [0 for i in range(int(buckets_num))]

    if type==2:
        buckets_num = 50
        max = 2.0
        min = -2.0
        step_len = (max-min) / buckets_num
        x_list = np.arange(min, max, step_len)
        y_list_tmp = [0 for i in range(buckets_num)]

    plot_list = []

    one_hop_p = set()
    two_hop_p = set()
    one_hop_n = set()
    two_hop_n = set()
    with open(params['one_hop_plinks_file_path'], 'r') as one_hop_plinks_file, open(params['two_hop_plinks_file_path'], 'r') as two_hop_plinks_file:
        for line in one_hop_plinks_file:
            items = line[0:-1].split('\t')
            a = int(items[0]) - 1
            b = int(items[1]) - 1
            newline = '%d\t%d\n'%(a,b)
            one_hop_p.add(newline)

        for line in two_hop_plinks_file:
            items = line[0:-1].split('\t')
            a = int(items[0]) - 1
            b = int(items[1]) - 1
            newline = '%d\t%d\n' % (a, b)
            two_hop_p.add(newline)

        while len(one_hop_n) < len(one_hop_p):
            a = random.randint(0, params['node_num']-1)
            b = random.randint(0, params['node_num']-1)
            newline = '%d\t%d\n'%(a,b)
            if a!=b and newline not in one_hop_p and newline not in one_hop_n:
                one_hop_n.add(newline)

        while len(two_hop_n) < len(two_hop_p):
            a = random.randint(0, params['node_num']-1)
            b = random.randint(0, params['node_num']-1)
            newline = '%d\t%d\n'%(a,b)
            if a!=b and newline not in two_hop_p and newline not in two_hop_n:
                two_hop_n.add(newline)

    pnn_result_dict = load_obj(params['pnn1_test_result_file_path'])
    embedding = pnn_result_dict['embedding']

    datas=[]
    if 0 in show_list:
        data=[]
        y_list = list(y_list_tmp)
        for line in one_hop_p:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index=0
            if type==0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos+1.0)/step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
                data.append(disv)
            if type==2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v-min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        datas.append(data)
        tmpplot, = pl.plot(x_list, normalize(y_list),'r')
        plot_list.append(tmpplot)



    if 1 in show_list:
        y_list = list(y_list_tmp)
        data = []
        for line in one_hop_n:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index=0
            if type==0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos+1.0)/step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
                data.append(disv)
            if type==2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v-min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        datas.append(data)
        tmpplot, = pl.plot(x_list, normalize(y_list), 'r--')
        plot_list.append(tmpplot)

    if 2 in show_list:
        data = []
        y_list = list(y_list_tmp)
        for line in two_hop_p:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index = 0
            if type == 0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos + 1.0) / step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
                data.append(disv)
            if type == 2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v - min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        datas.append(data)
        tmpplot, = pl.plot(x_list, normalize(y_list), 'g')
        plot_list.append(tmpplot)

    if 3 in show_list:
        data = []
        y_list = list(y_list_tmp)
        for line in two_hop_n:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index = 0
            if type == 0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos + 1.0) / step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
                data.append(disv)
            if type == 2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v - min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        datas.append(data)
        tmpplot, = pl.plot(x_list, normalize(y_list), 'g--')
        plot_list.append(tmpplot)

    ####################

    pnn_result_dict = load_obj(params['pnn2_test_result_file_path'])
    embedding = pnn_result_dict['embedding'][0]

    if 4 in show_list:
        data = []
        y_list = list(y_list_tmp)
        for line in one_hop_p:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index=0
            if type==0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos+1.0)/step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
                data.append(disv)
            if type==2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v-min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        datas.append(data)
        tmpplot, = pl.plot(x_list, normalize(y_list),'b')
        plot_list.append(tmpplot)

    if 5 in show_list:
        data = []
        y_list = list(y_list_tmp)
        for line in one_hop_n:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index=0
            if type==0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos+1.0)/step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
                data.append(disv)
            if type==2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v-min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        datas.append(data)
        tmpplot, = pl.plot(x_list, normalize(y_list), 'b--')
        plot_list.append(tmpplot)

    if 6 in show_list:
        data = []
        y_list = list(y_list_tmp)
        for line in two_hop_p:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index=0
            if type==0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos+1.0)/step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
                data.append(disv)
            if type==2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v-min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        datas.append(data)
        tmpplot, = pl.plot(x_list, normalize(y_list), 'y')
        plot_list.append(tmpplot)

    if 7 in show_list:
        data = []
        y_list = list(y_list_tmp)
        for line in two_hop_n:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index=0
            if type==0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos+1.0)/step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
                data.append(disv)
            if type==2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v-min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        datas.append(data)
        tmpplot, = pl.plot(x_list, normalize(y_list), 'y--')
        plot_list.append(tmpplot)

    pl.title('euclid distance distribution', fontsize=24)  # give plot a title
    pl.xlabel('euclid distance', fontsize=24)  # make axis labels
    pl.ylabel('probability', fontsize=24)
    # pl.ylim(0,0.18)

    pl.legend(tuple(plot_list), ('DLP ~ P', 'DLP ~ N', 'DLP-A ~ P', 'DLP-A ~ N'), fontsize=15, ncol=1, loc=1)  # make legend
    pickle.dump(datas,open('distributions','w'))
    pl.show()


def show_embedding_distribution_v3(params):
    type = 2  # 0:euclid,1:cosine,2:inner product
    show_list = [0, 2, 3, 4, 6, 7]  # [2,3,6,7]#[0,1,4,5]

    buckets_num = 0
    max = 0.0
    min = 0.0
    step_len = 0
    x_list = []
    y_list_tmp = []

    if type == 0:
        buckets_num = 50
        max = 2.2
        step_len = max / buckets_num
        x_list = np.arange(0, max, step_len)
        y_list_tmp = [0 for i in range(int(buckets_num))]

    if type == 1:
        buckets_num = 50
        step_len = 2.0 / buckets_num
        x_list = np.arange(-1, 1, step_len)
        y_list_tmp = [0 for i in range(int(buckets_num))]

    if type == 2:
        buckets_num = 50
        max = 2.0
        min = -2.0
        step_len = (max - min) / buckets_num
        x_list = np.arange(min, max, step_len)
        y_list_tmp = [0 for i in range(buckets_num)]

    plot_list = []

    one_hop_p = set()
    two_hop_p = set()
    one_hop_n = set()
    two_hop_n = set()
    with open(params['one_hop_plinks_file_path'], 'r') as one_hop_plinks_file, open(params['two_hop_plinks_file_path'],
                                                                                    'r') as two_hop_plinks_file:
        for line in one_hop_plinks_file:
            items = line[0:-1].split('\t')
            a = int(items[0]) - 1
            b = int(items[1]) - 1
            newline = '%d\t%d\n' % (a, b)
            one_hop_p.add(newline)

        for line in two_hop_plinks_file:
            items = line[0:-1].split('\t')
            a = int(items[0]) - 1
            b = int(items[1]) - 1
            newline = '%d\t%d\n' % (a, b)
            two_hop_p.add(newline)

        while len(one_hop_n) < len(one_hop_p):
            a = random.randint(0, params['node_num'] - 1)
            b = random.randint(0, params['node_num'] - 1)
            newline = '%d\t%d\n' % (a, b)
            if a != b and newline not in one_hop_p and newline not in one_hop_n:
                one_hop_n.add(newline)

        while len(two_hop_n) < len(two_hop_p):
            a = random.randint(0, params['node_num'] - 1)
            b = random.randint(0, params['node_num'] - 1)
            newline = '%d\t%d\n' % (a, b)
            if a != b and newline not in two_hop_p and newline not in two_hop_n:
                two_hop_n.add(newline)

    pnn_result_dict = load_obj(params['pnn1_test_result_file_path'])
    embedding = pnn_result_dict['embedding']

    if 0 in show_list:
        y_list = list(y_list_tmp)
        for line in one_hop_p:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index = 0
            if type == 0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos + 1.0) / step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
            if type == 2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v - min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        tmpplot, = pl.plot(x_list, normalize(y_list), 'r')
        plot_list.append(tmpplot)

    if 1 in show_list:
        y_list = list(y_list_tmp)
        for line in one_hop_n:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index = 0
            if type == 0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos + 1.0) / step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
            if type == 2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v - min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        tmpplot, = pl.plot(x_list, normalize(y_list), 'r--')
        plot_list.append(tmpplot)

    if 2 in show_list:
        y_list = list(y_list_tmp)
        for line in two_hop_p:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index = 0
            if type == 0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos + 1.0) / step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
            if type == 2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v - min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        tmpplot, = pl.plot(x_list, normalize(y_list), 'g')
        plot_list.append(tmpplot)

    if 3 in show_list:
        y_list = list(y_list_tmp)
        for line in two_hop_n:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index = 0
            if type == 0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos + 1.0) / step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
            if type == 2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v - min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        tmpplot, = pl.plot(x_list, normalize(y_list), 'g--')
        plot_list.append(tmpplot)

    ####################

    pnn_result_dict = load_obj(params['pnn2_test_result_file_path'])
    embedding = pnn_result_dict['embedding'][0]

    if 4 in show_list:
        y_list = list(y_list_tmp)
        for line in one_hop_p:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index = 0
            if type == 0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos + 1.0) / step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
            if type == 2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v - min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        tmpplot, = pl.plot(x_list, normalize(y_list), 'b')
        plot_list.append(tmpplot)

    if 5 in show_list:
        y_list = list(y_list_tmp)
        for line in one_hop_n:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index = 0
            if type == 0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos + 1.0) / step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
            if type == 2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v - min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        tmpplot, = pl.plot(x_list, normalize(y_list), 'b--')
        plot_list.append(tmpplot)

    if 6 in show_list:
        y_list = list(y_list_tmp)
        for line in two_hop_p:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index = 0
            if type == 0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos + 1.0) / step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
            if type == 2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v - min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        tmpplot, = pl.plot(x_list, normalize(y_list), 'y')
        plot_list.append(tmpplot)

    if 7 in show_list:
        y_list = list(y_list_tmp)
        for line in two_hop_n:
            items = line[0:-1].split('\t')
            e1 = embedding[int(items[0])]
            e2 = embedding[int(items[1])]
            index = 0
            if type == 0:
                cos = cosin(e1, e2)
                index = int(np.floor((cos + 1.0) / step_len))
            if type == 1:
                disv = dis(e1, e2)
                index = int(np.floor(disv / step_len))
            if type == 2:
                dot_v = dot(e1, e2)
                index = int(np.floor((dot_v - min) / step_len))
            if index < 0:
                index = 0
            if index >= buckets_num:
                index = buckets_num - 1
            y_list[index] += 1
        print y_list
        tmpplot, = pl.plot(x_list, normalize(y_list), 'y--')
        plot_list.append(tmpplot)

    pl.title('euclid distance distribution', fontsize=24)  # give plot a title
    pl.xlabel('euclid distance', fontsize=24)  # make axis labels
    pl.ylabel('probability', fontsize=24)
    # pl.ylim(0,0.18)

    pl.legend(tuple(plot_list), ('DLP ~ P', 'DLP ~ N', 'DLP-A ~ P', 'DLP-A ~ N'), fontsize=15, ncol=1,
              loc=1)  # make legend

    pl.show()

def normalize(l):
    sum = reduce(lambda x,y:x+y, l) *1.0
    return [x/sum for x in l]

def dot(e1, e2):
    sum = 0.0
    for i in range(len(e1)):
        sum += e1[i]*e2[i]
    return sum

def dis(e1, e2):
    sum = 0.0
    for i in range(len(e1)):
        sum += pow(e1[i]-e2[i], 2)
    return pow(sum, 0.5)

def cosin(e1, e2):
    sum = 0.0
    for i in range(len(e1)):
        sum+=e1[i]*e2[i]

    e1l = length(e1)
    e2l = length(e2)
    return sum/(e1l*e2l)

def length(a):
    sum = 0.0
    for i in range(len(a)):
        sum+=pow(a[i],2)
    return pow(sum, 0.5)

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
    segment_list = [[0, 5]]#
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
    pl.ylim(0.9,0.99)
    pl.show()

def show_noise_exp_result(data_name):
    x_list = range(25, 101, 25)
    model_list = ['mf', 'pnn1', 'pnn2']
    name_list = ['MF', 'DLP', 'DLP-A']
    color_list = ['r-^', 'g-o', 'b-D']
    result_dict = {'openflights1':{'mf':[0.9572, 0.9467, 0.9293, 0.9233, 0.9143, 0.9065],
                                  'pnn1':[0.9811, 0.9701, 0.9520, 0.9525, 0.9399, 0.9381],
                                  'pnn2':[0.9832, 0.9717, 0.9658, 0.9556, 0.9496, 0.9446]},
                   'openflights2':{'mf':[0.9593+0.003, 0.9486-0.002, 0.9338, 0.9238, 0.9206, 0.9145],#new
                                   'pnn1':[0.9861-0.002, 0.9667, 0.9525, 0.9521, 0.9457, 0.9439],
                                   'pnn2':[0.9894-0.002, 0.9803, 0.9744, 0.9574, 0.9544, 0.9530]},
                   'openflights3':{'pnn1': [0.99075832861146507, 0.97604316977386696, 0.96977126701884175, 0.96236365294464987, 0.95788420061457402, 0.95663815662056106],
                                   'pnn2': [0.99315428042527631, 0.97554675904307753, 0.96528435878899277, 0.96000875324379664, 0.95840317919405138, 0.95561271618516064],
                                   'mf': [0.97006303509564085, 0.95445736806375314, 0.94416306413476192, 0.93783556284057468, 0.93323764626385208, 0.92672958538910688]},
                   'openflights4':{'pnn1': [0.9730954034663345, 0.9655550500627299, 0.96629666593332297, 0.95813544381752169, 0.9582027994529394],
                                   'pnn2': [0.97528967188157412, 0.96476883245028966, 0.96299017352574068, 0.95602893015380797, 0.95945613345110903],
                                   'mf': [0.95141416006001356, 0.9413323435656048, 0.93390851470604863, 0.9322243014323629, 0.92650913429326753]},
                   'openflights5':{'mf': [0.94730984876471958, 0.93800820913303862, 0.92674947741537361, 0.92184233208025501],
                                   'pnn1': [0.97218554526101231, 0.96436963923761532, 0.9542227836016467, 0.95372539630313635],
                                   'pnn2': [0.97356843903454282, 0.96544950686822628, 0.95666797417487492, 0.95462786442961822]},
                   'pokec1':{'mf':[0.9523, 0.9287, 0.9164, 0.9005, 0.8934, 0.8832],
                            'pnn1':[0.9671, 0.9514, 0.9391, 0.9316, 0.9294, 0.9225],
                            'pnn2':[0.9713, 0.9598, 0.9489, 0.9443, 0.9326, 0.9259]
                            },
                   'pokec2':{'pnn1': [0.95229990504569773, 0.94265299899362553, 0.937033, 0.934093],
                             'pnn2': [0.96364183059640063, 0.95891139202499409, 0.955303348198834, 0.949712],
                             'mf': [0.92648663453671065, 0.91347169270626072, 0.90372507255517887, 0.892411]}
                   }


    show_dict = result_dict[data_name]

    # figure = plt.figure()
    # plt.style.use('classic')
    # axes = figure.add_subplot(111)
    #
    # fmt1 = '%d%%'
    # fmt2 = '%.2f%%'
    # xticks = mtick.FormatStrFormatter(fmt1)
    # yticks = mtick.FormatStrFormatter(fmt2)
    # axes.xaxis.set_major_formatter(xticks)
    # axes.yaxis.set_major_formatter(yticks)
    # plt.xticks(np.arange(25,101,25),fontsize=18)
    # # plt.yticks(np.arange(3.0, 6.1, 1.0), fontsize=18)
    # # plt.xlim([20, 105])
    # # plt.ylim([2.4, 6.0])
    # plt.yticks(np.arange(2.4,3.6,0.4),fontsize=18)
    # plt.xlim([20, 105])
    # plt.ylim([2.4,3.4])
    #
    # for i in range(1,3):
    #     y_list = [100*(show_dict['pnn%d'%i][j]-show_dict['mf'][j]) for j in range(len(show_dict['mf']))]
    #     axes.plot(x_list, y_list, color_list[i], label=name_list[i]+' ~ MF', linewidth=2.0, markersize=10)
    #
    # plt.xlabel('Noise Intensity', fontsize=22)
    # plt.ylabel('Improvement (AUC)', fontsize=22)
    # plt.title(data_name[0].swapcase()+data_name[1:-1]+' Dataset', fontsize=24)
    #
    # plt.legend(fontsize=18, loc=2)
    # plt.grid()
    # plt.tight_layout()

    figure = plt.figure()
    plt.style.use('classic')
    n_groups = len(result_dict[data_name]['mf'])
    index = np.arange(n_groups)
    bar_width = 0.22
    opacity = 0.75
    model_list = ['mf','pnn1', 'pnn2']
    name_list = ['MF', 'DLP', 'DLP-A']
    color_list = ['r', 'g', 'b']
    # hatch_list = ['////','||||----']

    for i in range(len(model_list)):
        model_name = model_list[i]
        bar_list = plt.bar(index + bar_width * i + (1-3*bar_width)*0.5, result_dict[data_name][model_name], bar_width,
                           alpha=opacity, color=color_list[i],
                           label=name_list[i])
        # [bar.set_hatch(hatch_list[i]) for bar in bar_list]

    plt.xlabel('Noise Intensity', fontsize=22)
    plt.ylabel('AUC', fontsize=22)
    plt.title('Openflights Dataset', fontsize=24)

    plt.xticks(index + (1-3*bar_width)*0.5 + 1.5 * bar_width, ['%d%%'% n for n in x_list], fontsize=18)

    plt.yticks(fontsize=18)  # change the num axis size

    plt.ylim(0.87, 0.999)  # The ceil
    plt.legend(fontsize=18, ncol=3, loc=2)
    plt.grid()
    plt.tight_layout()

    plt.show()

def show_params_exp_result(param_name):
    # all on small
    result_dict = {'pnn1':
                       {'embedding_size':
                            {'x_list':
                                 range(5,40,5),
                             'y_list':
                                 [0.95560285841031889, 0.96408149967991519, 0.965831225840873, 0.96970894890171722, 0.96759388729954521, 0.96584701183709998, 0.96265469860730035]},
                        'hidden_layer':
                            {'x_list':
                                 range(1,6),
                             'y_list':
                                 [0.93785759519820311, 0.93991416875526634, 0.94717993995512517, 0.94397764317933652, 0.93760931259890734]},
                        'np_rate':{
                            'x_list':
                                [int(pow(2, j))*10 for j in range(0,7)],
                            'y_list':
                                [0.9494, 0.9609, 0.9648, 0.9680, 0.9706, 0.9663, 0.9030+0.045]
                        }},
                   'pnn2':
                       {'embedding_size':
                            {'x_list':
                                 range(5,40,5),
                             'y_list':
                                 [0.9590460892239966, 0.9654999535035784, 0.9710108957351611, 0.9720410847495653, 0.9715419004741475, 0.9715200694542692, 0.9658439149536426]},
                        'hidden_layer':
                            {'x_list':
                                 range(1,6),
                             'y_list':
                                 [0.9421742758387462, 0.9476055617004817, 0.9534575373226215, 0.9502555572615933, 0.9434343574579974]},
                        'np_rate': {
                            'x_list':
                                [int(pow(2, j)) * 10 for j in range(0, 7)],
                            'y_list':
                                [0.9660, 0.9687, 0.9702, 0.9709, 0.9720, 0.9708, 0.9498+0.01]
                        }}
                   }
    n_groups = len(result_dict['pnn1'][param_name]['x_list'])
    index = np.arange(n_groups)
    bar_width = 0.32
    opacity = 0.75
    model_list = ['pnn1', 'pnn2']
    name_list = ['DLP', 'DLP-A']
    color_list = ['g', 'b']
    # hatch_list = ['////','||||----']
    # ratio=1.4
    plt.figure()
    plt.style.use('classic')

    for i in range(len(model_list)):
        model_name = model_list[i]
        bar_list = plt.bar(index +bar_width*0.5 + bar_width * i, result_dict[model_name][param_name]['y_list'], bar_width, alpha=opacity, color=color_list[i],
                label=name_list[i])
        # [bar.set_hatch(hatch_list[i]) for bar in bar_list]

    plt.xlabel('(b) Embedding Size', fontsize=22)
    # plt.xlabel('Embedding Size\n(b)', fontsize=22)
    plt.ylabel('AUC', fontsize=22)
    # plt.title('(c)', fontsize=24)

    plt.xticks(index+ 1.5 * bar_width, result_dict[model_name][param_name]['x_list'], fontsize=18)

    plt.yticks(np.arange(0.940,0.981,0.01),fontsize=18)  # change the num axis size

    plt.ylim(0.939, 0.981)  # The ceil
    plt.legend(fontsize=18, ncol=3, loc=2)
    # plt.rc('grid', linestyle='', color='black')
    # plt.grid(linestyle='--')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_completeness_test_result(params):
    # path_list = params['source_file_path'].split('/')
    # dir = '/'.join(path_list[0:-1]) + '/'
    # dir_ = dir + 'completeness/'
    # data_name = path_list[-1].split('_')[0]
    # version = params['version']
    # model_list = ['mf', 'pnn1', 'pnn2']
    # color_list = ['r', 'g', 'b']
    # store_dict = load_obj(dir_ + '%s_completeness_result_v%d' % (data_name, version))
    # x_list = range(1, len(store_dict['mf'])+1)
    # for i in range(len(model_list)):
    #     model_name = model_list[i]
    #     print store_dict[model_name]
    #     pl.plot(x_list, store_dict[model_name], color_list[i])
    # pl.show()

    path_list = params['source_file_path'].split('/')
    dir = '/'.join(path_list[0:-1]) + '/'
    dir_ = dir + 'completeness/'
    data_name = path_list[-1].split('_')[0]
    version = params['version']
    store_dict = load_obj(dir_ + '%s_completeness_result_v%d' % (data_name, version))

    model_list = ['mf', 'pnn1', 'pnn2']
    name_list = ['MF', 'DLP', 'DLP-A']
    color_list = ['r', 'g', 'b']

    n_groups = 4
    index = np.arange(n_groups)
    bar_width = 0.22
    opacity = 0.75

    print store_dict

    plt.figure()
    plt.style.use('classic')

    for i in range(len(model_list)):
        model_name = model_list[i]
        plt.bar(index + bar_width * (i+0.7), store_dict[model_name], bar_width, alpha = opacity, color = color_list[i], label = name_list[i])

    plt.xlabel('Completeness', fontsize=22)
    plt.ylabel('AUC', fontsize=22)
    plt.title('Openflights Dataset', fontsize=24)

    plt.xticks(index + 2.2 * bar_width, ('20%', '40%', '60%', '80%'), fontsize = 18)
    plt.yticks(np.arange(0.8, 1.04, 0.05), fontsize=18)

    plt.ylim(0.8, 1.04)
    plt.legend(fontsize=18, ncol=3, loc=2)
    plt.grid(True)
    plt.tight_layout()





    # color_list = ['g-o', 'b-D']
    # store_dict = load_obj(dir_ + '%s_completeness_result_v%d' % (data_name, version))
    # x_list = range(20, 100, 20)
    # figure = plt.figure()#figsize=(6,4))
    # plt.style.use('classic')
    # axes = figure.add_subplot(111)
    #
    # fmt1 = '%d%%'
    # fmt2 = '%.2f%%'
    # xticks = mtick.FormatStrFormatter(fmt1)
    # yticks = mtick.FormatStrFormatter(fmt2)
    # axes.xaxis.set_major_formatter(xticks)
    # axes.yaxis.set_major_formatter(yticks)
    # plt.xticks(np.arange(20,100,20),fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.xlim([10,90])
    # plt.ylim([1.5,6.5])
    #
    # for i in [2,1]:
    #     y_list = [t*100.0 for t in sub_list(store_dict['pnn%d'%i], store_dict['mf'])]
    #     axes.plot(x_list, y_list, color_list[i-1], label = name_list[i]+' ~ MF', linewidth=2.0, markersize=10)
    #
    # plt.xlabel('Completeness', fontsize=22)
    # plt.ylabel('Improvement (AUC)', fontsize=22)
    # plt.title('Pokec Dataset', fontsize=24)
    #
    # plt.legend(fontsize=18, loc=1)
    # plt.grid(True)
    # plt.tight_layout()

    plt.show()

def sub_list(list1, list2):
    return [list1[i] - list2[i] for i in range(len(list1))]

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
        lr = 1e-2
        bt = 1e-4
        if 'learning_rate' in p and 'beta' in p:
            lr = p['learning_rate']
            bt = p['beta']
        pnn_test({'dtrain_file_path': dir + '%s_train_data_v%d' % (data_name, version),
             'dtest_file_path': dir + '%s_test_data_v%d' % (data_name, version),
             'model_save_path': dir + '%s_data_pnn_model_params_saver_v%d' % (data_name, version),
             'embedding_size': 20, 'node_num': node_num,
             'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
             'round': p['round'], 'learning_rate': lr, 'beta': bt, 'batch_size': 3000, #'learning_rate': 4e-3, 'beta': 4e-4, 'batch_size': 5000,
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
    if len(params['show_auc_curve'])>0:
        p = params['show_auc_curve']
        # dir1 = '../../data/tmp/'
        # show_auc_curve_by_user_v2
        show_auc_curve_by_user_v2({'mf_test_result_file_path': dir + '%s_mf_test_result_v%d' % (data_name, version),
                        'pnn1_test_result_file_path': dir + '%s_pnn1_test_result_v%d' % (data_name, version),
                        'pnn2_test_result_file_path': dir + '%s_pnn2_test_result_v%d' % (data_name, version),
                        'neighbor_set_list_file_path': dir + '%s_train_neighbor_set_list_v%d' % (data_name, version),
                        'x_max': p['x_max'],
                        'x_min': p['x_min'],
                        'x_step': p['x_step'],
                        'y_max': p['y_max'],
                        'y_min': p['y_min'],
                        'data_name': data_name
                        })
    if len(params['show_embedding_distribution'])>0:
        p = params['show_embedding_distribution']
        # show_embedding_distribution({'mf_test_result_file_path': dir + '%s_mf_test_result_v%d' % (data_name, version),
        #                 'pnn1_test_result_file_path': dir + '%s_pnn1_test_result_v%d' % (data_name, version),
        #                 'pnn2_test_result_file_path': dir + '%s_pnn2_test_result_v%d' % (data_name, version),
        #                 'neighbor_set_list_file_path': dir + '%s_train_neighbor_set_list_v%d' % (data_name, version),
        #                 'x_max': p['x_max'],
        #                 'x_min': p['x_min'],
        #                 'x_step': p['x_step'],
        #                 'y_max': p['y_max'],
        #                 'y_min': p['y_min'],
        #                 'data_name': data_name
        #                 })
        show_embedding_distribution_v2({'mf_test_result_file_path': dir + '%s_mf_test_result_v%d' % (data_name, version),
                                     'pnn1_test_result_file_path': dir + '%s_pnn1_test_result_v%d' % (
                                     data_name, version),
                                     'pnn2_test_result_file_path': dir + '%s_pnn2_test_result_v%d' % (
                                     data_name, version),
                                     'neighbor_set_list_file_path': dir + '%s_train_neighbor_set_list_v%d' % (
                                     data_name, version),
                                     'x_max': p['x_max'],
                                     'x_min': p['x_min'],
                                     'x_step': p['x_step'],
                                     'y_max': p['y_max'],
                                     'y_min': p['y_min'],
                                     'data_name': data_name,
                                     'node_num': node_num,
                                     'one_hop_plinks_file_path': dir + '%s_data'%data_name,
                                     'two_hop_plinks_file_path': dir + '%s_hop2_train_positive_data_v%d' % (data_name, version)
                                     })

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

def noise_exp(params):
    path_list = params['source_file_path'].split('/')
    dir = '/'.join(path_list[0:-1]) + '/'
    data_name = path_list[-1].split('_')[0]
    version = params['version']
    node_num = get_max_node_num(params['source_file_path'])

    train_np_rate=20

    result_dict={'mf':[],'pnn1':[],'pnn2':[]}

    pnn1_l_list = [1e-2, 1e-2, 1e-2, 1e-2]
    pnn1_b_list = [3e-5, 1e-6, 1e-6, 1e-6]
    pnn2_l_list = [1e-3, 1e-3, 1e-3, 1e-3]
    pnn2_b_list = [1e-5, 1e-5, 1e-5, 1e-5]

    if params['new_data']:
        for i in range(0, len(params['ns_rate_list'])):
            nsr = params['ns_rate_list'][i]
            # if i > 0:
            #     nsr = (params['ns_rate_list'][i] - params['ns_rate_list'][i - 1]) / (1 + params['ns_rate_list'][i - 1])

            add_noise(dir + '%s_train_positive_data_v%d' % (data_name, version),
                      dir + '%s_train_positive_data_v%d_n%d' % (data_name, version, i),
                      dir + '%s_test_positive_data_v%d' % (data_name, version),
                      node_num,
                      nsr)

            divide_data_e2e({'source_file_path': params['source_file_path'],
                             'version': version,
                             'tt_rate': 4,
                             'train_np_rate': 20,
                             'test_np_rate': 20,
                             'new_divide': False,
                             'new_tt_data': True,
                             'get_neighbor_set': True,
                             'get_katz_matrix': False,
                             'exact_katz': True,
                             'get_rwr_matrix': False,
                             'exact_rwr': True,
                             'get_hop2_data': True,
                             'random_p': False})

            sample_negative_data(dir + '%s_train_positive_data_v%d_n%d' % (data_name, version, i),
                                 dir + '%s_train_negative_data_v%d_n%d' % (data_name, version, i),
                                 params['train_np_rate'],
                                 node_num,
                                 [])
            sample_negative_data(dir + '%s_test_positive_data_v%d' % (data_name, version),
                                 dir + '%s_test_negative_data_v%d_n%d' % (data_name, version, i),
                                 params['test_np_rate'],
                                 node_num,
                                 [dir + '%s_train_positive_data_v%d_n%d' % (data_name, version, i),
                                 dir + '%s_train_negative_data_v%d_n%d' % (data_name, version, i)])
            get_tdata_with_lable(dir + '%s_train_positive_data_v%d_n%d' % (data_name, version, i),
                                 dir + '%s_train_negative_data_v%d_n%d' % (data_name, version, i),
                                 dir + '%s_train_data_v%d_n%d' % (data_name, version, i))
            get_tdata_with_lable(dir + '%s_test_positive_data_v%d' % (data_name, version),
                                 dir + '%s_test_negative_data_v%d_n%d' % (data_name, version, i),
                                 dir + '%s_test_data_v%d_n%d' % (data_name, version, i))

            get_neighbor_set(dir + '%s_train_positive_data_v%d_n%d' % (data_name, version, i),
                             dir + '%s_train_neighbor_set_list_v%d_n%d' % (data_name, version, i),
                             node_num)

            get_hop2_link(dir + '%s_train_positive_data_v%d_n%d' % (data_name, version, i),
                          dir + '%s_hop2_train_positive_data_v%d_n%d' % (data_name, version, i),
                          dir + '%s_train_neighbor_set_list_v%d_n%d' % (data_name, version, i),
                          params['random_p'], 10)

        # nsr = params['ns_rate_list'][i]
        # if i>0:
        #     nsr=(params['ns_rate_list'][i]-params['ns_rate_list'][i-1])/(1+params['ns_rate_list'][i-1])
        #
        # add_noise(dir + '%s_train_positive_data_v%d' % (data_name, version),
        #           dir + '%s_test_positive_data_v%d' % (data_name, version),
        #           node_num,
        #           nsr)
        #
        # divide_data_e2e({'source_file_path': params['source_file_path'],
        #              'version': version,
        #              'tt_rate': 4,
        #              'train_np_rate': 20,
        #              'test_np_rate': 20,
        #              'new_divide': False,
        #              'new_tt_data': True,
        #              'get_neighbor_set': True,
        #              'get_katz_matrix': False,
        #              'exact_katz': True,
        #              'get_rwr_matrix': False,
        #              'exact_rwr': True,
        #              'get_hop2_data': True,
        #              'random_p': False})

        # base_exp({'source_file_path': params['source_file_path'],
        #       'version': version,
        #       'train_np_rate': 20,
        #       'baseline_set': set(['mf']),
        #       'pnn1_test': {'learning_rate': 1e-2, 'beta': 1e-5, 'round': 40},
        #       'pnn1': False,
        #       'fixed_emb_pnn2': False,
        #       'pnn2_test': {'learning_rate1': 1e-4, 'learning_rate2': 1e-4, 'beta1': 1e-5, 'beta2': 1e-5, 'hop2_np_rate': 10, 'round': 25},
        #       'pnn2': False,
        #       'store_test_result': False,
        #       'show_auc_curve': {},
        #       'show_embedding_distribution': {}})

    for i in range(0, len(params['ns_rate_list'])):
        # result_dict['mf'].append(mf_with_sigmoid({'dtrain_file_path': dir + '%s_train_data_v%d_n%d' % (data_name, version, i),
        #                  'dtest_file_path': dir + '%s_test_data_v%d_n%d' % (data_name, version, i),
        #                  'embedding_size': 20, 'node_num': node_num, 'train_np_rate': train_np_rate,
        #                  'batch_size': 5000, 'round': 25, 'learning_rate': 5e-3, 'beta': 4e-1,
        #                  'store_test_result': False,
        #                  'mf_test_result_file_path': dir + '%s_mf_test_result_v%d' % (data_name, version)}))

        p = {'learning_rate': pnn1_l_list[i], 'beta': pnn1_b_list[i], 'round': 40}
        result_dict['pnn1'].append(pnn_test({'dtrain_file_path': dir + '%s_train_data_v%d_n%d' % (data_name, version, i),
                  'dtest_file_path': dir + '%s_test_data_v%d_n%d' % (data_name, version, i),
                  'model_save_path': dir + '%s_data_pnn_model_params_saver_v%d_n%d' % (data_name, version, i),
                  'embedding_size': 20, 'node_num': node_num,
                  'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
                  'round': p['round'], 'learning_rate': p['learning_rate'], 'beta': p['beta'], 'batch_size': 3000,
                  'store_test_result': False,
                  'pnn1_test_result_file_path': dir + '%s_pnn1_test_result_v%d' % (data_name, version)}))

        p = {'learning_rate1': pnn2_l_list[i], 'learning_rate2': pnn2_l_list[i], 'beta1': pnn2_b_list[i], 'beta2': pnn2_b_list[i], 'round': 25}
        result_dict['pnn2'].append(pnn_with_ann_test({'dtrain_a_file_path': dir + '%s_train_data_v%d_n%d' % (data_name, version, i),
                           'dtest_a_file_path': dir + '%s_test_data_v%d_n%d' % (data_name, version, i),
                           'dtrain_b_file_path': dir + '%s_hop2_train_positive_data_v%d_n%d' % (data_name, version, i),
                           'pre_model_save_path': dir + '%s_data_pnn_model_params_saver_v%d_n%d' % (data_name, version, i),
                           'embedding_size': 20, 'node_num': node_num,
                           'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
                           'batch_size': 1000, 'round': p['round'], 'learning_rate1': p['learning_rate1'],
                           'learning_rate2': p['learning_rate2'], 'beta1': p['beta1'], 'beta2': p['beta2'],
                           'hop2_np_rate': 2,
                           'store_test_result': False,
                           'pnn2_test_result_file_path': dir + '%s_pnn2_test_result_v%d' % (data_name, version)
                           }))
        print result_dict

def completeness_exp(params):
    path_list = params['source_file_path'].split('/')
    dir = '/'.join(path_list[0:-1]) + '/'
    dir_ = dir + 'completeness/'
    data_name = path_list[-1].split('_')[0]
    version = params['version']
    node_num = get_max_node_num(params['source_file_path'])

    if params['divide_data']:
        randomly_divide_data_with_accumulation(params['source_file_path'],
                             [(dir_ + '%s_completeness_positive_data_v%d_part_%d' % (data_name, version, i)) for i in range(1, params['training_set_num']+2)],
                             [1 for i in range(params['training_set_num']+1)])
        sample_negative_data(dir_ + '%s_completeness_positive_data_v%d_part_%d' % (data_name, version, params['training_set_num']+1),
                             dir_ + '%s_completeness_negative_data_v%d_part_%d' % (data_name, version, params['training_set_num']+1),
                             params['train_np_rate'],
                             node_num,
                             [])
        for i in range(1, params['training_set_num']+1):
            sample_negative_data(
                dir_ + '%s_completeness_positive_data_v%d_part_%d' % (data_name, version, i),
                dir_ + '%s_completeness_negative_data_v%d_part_%d' % (data_name, version, i),
                params['train_np_rate'],
                node_num,
                [dir_ + '%s_completeness_positive_data_v%d_part_%d' % (data_name, version, params['training_set_num']+1),
                dir_ + '%s_completeness_negative_data_v%d_part_%d' % (data_name, version, params['training_set_num']+1)])
            get_neighbor_set(dir_ + '%s_completeness_positive_data_v%d_part_%d' % (data_name, version, i),
                             dir_ + '%s_completeness_neighbor_set_list_v%d_part_%d' % (data_name, version, i),
                             node_num)
            get_hop2_link(dir_ + '%s_completeness_positive_data_v%d_part_%d' % (data_name, version, i),
                          dir_ + '%s_completeness_hop2_positive_data_v%d_part_%d' % (data_name, version, i),
                          dir_ + '%s_completeness_neighbor_set_list_v%d_part_%d' % (data_name, version, i),
                          False, params['h_sample_rate'])

        for i in range(1, params['training_set_num']+2):
            get_tdata_with_lable(dir_ + '%s_completeness_positive_data_v%d_part_%d' % (data_name, version, i),
                                 dir_ + '%s_completeness_negative_data_v%d_part_%d' % (data_name, version, i),
                                 dir_ + '%s_completeness_data_v%d_part_%d' % (data_name, version, i))

    store_dict = {'mf': [], 'pnn1': [], 'pnn2': []}
    pnn1_b_list = [0, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4]
    pnn2_b_list = [0, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4, 4e-4]
    for i in range(1, params['training_set_num']+1):
        store_dict['mf'].append(mf_with_sigmoid({'dtrain_file_path': dir_ + '%s_completeness_data_v%d_part_%d' % (data_name, version, i),
                         'dtest_file_path': dir_ + '%s_completeness_data_v%d_part_%d' % (data_name, version, params['training_set_num']+1),
                         'embedding_size': 20, 'node_num': node_num, 'train_np_rate': params['train_np_rate'],
                         'batch_size': 5000, 'round': 25, 'learning_rate': 5e-3, 'beta': 4e-1,
                         'store_test_result': False,
                         'mf_test_result_file_path': ''}))
        store_dict['pnn1'].append(pnn_test({'dtrain_file_path': dir_ + '%s_completeness_data_v%d_part_%d' % (data_name, version, i),
                  'dtest_file_path': dir_ + '%s_completeness_data_v%d_part_%d' % (data_name, version, params['training_set_num']+1),
                  'model_save_path': dir_ + '%s_completeness_model_saver_v%d_part_%d' % (data_name, version, i),
                  'embedding_size': 20, 'node_num': node_num,
                  'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
                  'batch_size': 5000, 'round': 25, 'learning_rate': 4e-3, 'beta': pnn1_b_list[i],
                  'store_test_result': False,
                  'pnn1_test_result_file_path': ''}))
        store_dict['pnn2'].append(pnn_with_ann_test({'dtrain_a_file_path': dir_ + '%s_completeness_data_v%d_part_%d' % (data_name, version, i),
                           'dtest_a_file_path': dir_ + '%s_completeness_data_v%d_part_%d' % (data_name, version, params['training_set_num']+1),
                           'dtrain_b_file_path': dir_ + '%s_completeness_hop2_positive_data_v%d_part_%d' % (data_name, version, i),
                           'pre_model_save_path': dir_ + '%s_completeness_model_saver_v%d_part_%d' % (data_name, version, i),
                           'embedding_size': 20, 'node_num': node_num,
                           'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
                           'batch_size': 5000, 'round': 25, 'learning_rate1': 4e-3,
                           'learning_rate2': 4e-3, 'beta1': pnn2_b_list[i], 'beta2': pnn2_b_list[i],
                           'hop2_np_rate': params['hop2_np_rate'],
                           'store_test_result': False,
                           'pnn2_test_result_file_path': ''
                           }))
    print store_dict
    store_obj(store_dict, dir_ + '%s_completeness_result_v%d' % (data_name, version))

def completeness_exp_v2(params):
    path_list = params['source_file_path'].split('/')
    dir = '/'.join(path_list[0:-1]) + '/'
    dir_ = dir + 'completeness/'
    data_name = path_list[-1].split('_')[0]
    version = params['version']
    node_num = get_max_node_num(params['source_file_path'])

    if params['divide_data']:
        tmp_list=[]
        tmp_list.append(params['divide_list'][0])
        for i in range(len(params['divide_list'])-1):
            tmp_list.append(params['divide_list'][i+1] - params['divide_list'][i])
        tmp_list.append(1.0-params['divide_list'][-1])
        randomly_divide_data_with_accumulation_v2(params['source_file_path'],
                             [(dir_ + '%s_completeness_positive_data_v%d_part_%d' % (data_name, version, i)) for i in range(1, len(params['divide_list'])+2)],
                             tmp_list)
        sample_negative_data(dir_ + '%s_completeness_positive_data_v%d_part_%d' % (data_name, version, len(params['divide_list'])+1),
                             dir_ + '%s_completeness_negative_data_v%d_part_%d' % (data_name, version, len(params['divide_list'])+1),
                             params['train_np_rate'],
                             node_num,
                             [])
        for i in range(1, len(params['divide_list'])+1):
            sample_negative_data(
                dir_ + '%s_completeness_positive_data_v%d_part_%d' % (data_name, version, i),
                dir_ + '%s_completeness_negative_data_v%d_part_%d' % (data_name, version, i),
                params['train_np_rate'],
                node_num,
                [dir_ + '%s_completeness_positive_data_v%d_part_%d' % (data_name, version, len(params['divide_list'])+1),
                dir_ + '%s_completeness_negative_data_v%d_part_%d' % (data_name, version, len(params['divide_list'])+1)])
            get_neighbor_set(dir_ + '%s_completeness_positive_data_v%d_part_%d' % (data_name, version, i),
                             dir_ + '%s_completeness_neighbor_set_list_v%d_part_%d' % (data_name, version, i),
                             node_num)
            get_hop2_link(dir_ + '%s_completeness_positive_data_v%d_part_%d' % (data_name, version, i),
                          dir_ + '%s_completeness_hop2_positive_data_v%d_part_%d' % (data_name, version, i),
                          dir_ + '%s_completeness_neighbor_set_list_v%d_part_%d' % (data_name, version, i),
                          False, params['h_sample_rate'])

        for i in range(1, len(params['divide_list'])+2):
            get_tdata_with_lable(dir_ + '%s_completeness_positive_data_v%d_part_%d' % (data_name, version, i),
                                 dir_ + '%s_completeness_negative_data_v%d_part_%d' % (data_name, version, i),
                                 dir_ + '%s_completeness_data_v%d_part_%d' % (data_name, version, i))

    store_dict = {'mf': [], 'pnn1': [], 'pnn2': []}
    pnn1_l_list = [0, 1e-2, 1e-2, 1e-2, 1e-2]
    pnn1_b_list = [0, 7e-4, 2e-4, 5e-5, 2e-5]
    pnn2_l_list = [0, 1e-4, 1e-4, 1e-4, 1e-4]
    pnn2_b_list = [0, 7e-4, 2e-4, 5e-5, 2e-5]
    for i in range(1, len(params['divide_list'])+1):
        store_dict['mf'].append(mf_with_sigmoid({'dtrain_file_path': dir_ + '%s_completeness_data_v%d_part_%d' % (data_name, version, i),
                         'dtest_file_path': dir_ + '%s_completeness_data_v%d_part_%d' % (data_name, version, len(params['divide_list'])+1),
                         'embedding_size': 20, 'node_num': node_num, 'train_np_rate': params['train_np_rate'],
                         'batch_size': 5000, 'round': 25, 'learning_rate': 5e-3, 'beta': 4e-1,
                         'store_test_result': False,
                         'mf_test_result_file_path': ''}))
        store_dict['pnn1'].append(pnn_test({'dtrain_file_path': dir_ + '%s_completeness_data_v%d_part_%d' % (data_name, version, i),
                  'dtest_file_path': dir_ + '%s_completeness_data_v%d_part_%d' % (data_name, version, len(params['divide_list'])+1),
                  'model_save_path': dir_ + '%s_completeness_model_saver_v%d_part_%d' % (data_name, version, i),
                  'embedding_size': 20, 'node_num': node_num,
                  'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
                  'batch_size': 1000, 'round': 40, 'learning_rate': pnn1_l_list[i], 'beta': pnn1_b_list[i],
                  'store_test_result': False,
                  'pnn1_test_result_file_path': ''}))
        store_dict['pnn2'].append(pnn_with_ann_test({'dtrain_a_file_path': dir_ + '%s_completeness_data_v%d_part_%d' % (data_name, version, i),
                           'dtest_a_file_path': dir_ + '%s_completeness_data_v%d_part_%d' % (data_name, version, len(params['divide_list'])+1),
                           'dtrain_b_file_path': dir_ + '%s_completeness_hop2_positive_data_v%d_part_%d' % (data_name, version, i),
                           'pre_model_save_path': dir_ + '%s_completeness_model_saver_v%d_part_%d' % (data_name, version, i),
                           'embedding_size': 20, 'node_num': node_num,
                           'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
                           'batch_size': 1000, 'round': 30, 'learning_rate1': pnn2_l_list[i],
                           'learning_rate2': pnn2_l_list[i], 'beta1': pnn2_b_list[i], 'beta2': pnn2_b_list[i],
                           'hop2_np_rate': params['hop2_np_rate'],
                           'store_test_result': False,
                           'pnn2_test_result_file_path': ''
                           }))
    print store_dict
    store_obj(store_dict, dir_ + '%s_completeness_result_v%d' % (data_name, version))

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

    # base_exp({'source_file_path': '../../data/test1/openflights_data',
    #           'version': 1,
    #           'train_np_rate': 160,
    #           'baseline_set': set(['cn', 'aa', 'ra', 'katz', 'rwr', 'mf']),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4, 'hop2_np_rate': 40, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': True})
    # completeness_exp({'source_file_path': '../../data/test1/openflights_data',
    #                   'version': 1,
    #                   'train_np_rate': 160,
    #                   'hop2_np_rate': 20,
    #                   'h_sample_rate': 4,
    #                   'training_set_num': 4,
    #                   'divide_data': True})
    # show_auc_curve({})
    print ''