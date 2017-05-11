import tensorflow as tf
import numpy as np
import pickle
import os
import math
import surprise
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from tfnmf import TFNMF
from random import randint
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from sklearn import metrics


#mf with nn
def exp0(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.truncated_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))
    q = tf.Variable(tf.truncated_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)

    # e_cc = tf.concat(1, [e1, e2])
    e_cc = e1*e2
    # e_cc = tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)), [-1, int(math.pow(params['embedding_size'], 2))])
    # e_cc = tf.concat(1, [e1, e2, e1*e2])
    # e_cc = tf.concat(1, [e1, e2, tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)), [-1, int(math.pow(params['embedding_size'], 2))])])
    # e_cc = tf.concat(1, [e1*e2, tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)), [-1, int(math.pow(params['embedding_size'], 2))])])
    # e_cc = tf.concat(1, [e1, e2, e1*e2, tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)), [-1, int(math.pow(params['embedding_size'], 2))])])


    weights1 = tf.Variable(tf.truncated_normal([params['embedding_size'], params['h1_size']], mean=0, stddev=0.01))
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.nn.relu(tf.matmul(e_cc, weights1) + biases1)

    weights2 = tf.Variable(tf.truncated_normal([params['h1_size'], params['h2_size']], mean=0, stddev=0.01))
    biases2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.nn.relu(tf.matmul(h1, weights2) + biases2)

    weights3 = tf.Variable(tf.truncated_normal([params['h2_size'], 1], mean=0, stddev=0.01))
    biases3 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.sigmoid(tf.matmul(h1, weights2) + biases2))
    ys_pre = tf.squeeze(tf.matmul(h2, weights3) + biases3)

    # loss = tf.reduce_mean(0.5 * tf.square(ys - ys_pre) + params['beta'] * (tf.reduce_sum(tf.square(e1) + tf.square(e2), 1)))
    weight_mss = tf.reduce_sum([tf.nn.l2_loss(weights1), tf.nn.l2_loss(weights2), tf.nn.l2_loss(weights3)])
    # biase_mss = tf.reduce_mean(
    #     [tf.reduce_mean(tf.square(biases1)), tf.reduce_mean(tf.square(biases2)), tf.reduce_mean(tf.square(biases3))])

    target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))
    loss = target_loss + params['beta'] * weight_mss

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)  # tell me why GradientDescentOptimizer did not work.........

    # rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - ys_pre)))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(ys_, tf.int32)), tf.float32))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        np.random.shuffle(dtrain)
        np.random.shuffle(dtest)
        data_size = dtrain.shape[0]
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i+1) % data_size
            if end <= start:
                continue
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run([train_step], feed_dict)
            if i%100 == 0:
                feed_dict = {u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])}
                train_rmse_v, train_ys_v, train_ys_pre_v, train_loss_v, train_target_loss_v = sess.run([rmse, ys, tf.sigmoid(ys_pre), loss, target_loss], feed_dict)
                feed_dict = {u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])}
                test_rmse_v, test_ys_v, test_ys_pre_v, test_loss_v, test_target_loss_v = sess.run([rmse, ys, tf.sigmoid(ys_pre), loss, target_loss], feed_dict)
                print 'step%5d: train<>rmse->%f, loss->%f, target_loss->%f, auc->%f------test<>rmse->%f, loss->%f, target_loss: %f, auc: %f' % (i, train_rmse_v, train_loss_v, train_target_loss_v, get_auc(train_ys_v, train_ys_pre_v), test_rmse_v, test_loss_v, test_target_loss_v, get_auc(test_ys_v, test_ys_pre_v))
        # y_trues, y_pres, accuracy_v = sess.run([ys, tf.sigmoid(ys_pre), accuracy], feed_dict = {u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        # print 'train auc: %f, train accuracy: %f' % (roc_auc_score(y_trues, y_pres), accuracy_v)
        # y_trues, y_pres, accuracy_v = sess.run([ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        # print 'test auc: %f, test accuracy: %f' % (roc_auc_score(y_trues, y_pres), accuracy_v)

#normal mf
def exp1(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
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
        print 'training...'
        rmse_v, loss_v, target_loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, target_loss, ys, ys_pre, accuracy],
                                                               feed_dict={u1s: dtrain[0:800000, 0],
                                                                          u2s: dtrain[0:800000, 1],
                                                                          ys: np.float32(dtrain[0:800000, 2])})
        t_rmse_v, t_loss_v, t_target_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, target_loss, ys, ys_pre, accuracy],
                                                                         feed_dict={u1s: dtest[0:800000, 0],
                                                                                    u2s: dtest[0:800000, 1],
                                                                                    ys: np.float32(dtest[0:800000, 2])})
        print '----step%5d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (
        0, rmse_v, loss_v, target_loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, t_target_loss_v, get_auc(t_y_trues, t_y_pres),
        t_accuracy_v)
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                rmse_v, loss_v, target_loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, target_loss, ys, ys_pre, accuracy],
                                                                       feed_dict={u1s: dtrain[0:800000, 0],
                                                                                  u2s: dtrain[0:800000, 1],
                                                                                  ys: np.float32(dtrain[0:800000, 2])})
                t_rmse_v, t_loss_v, t_target_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run(
                    [rmse, loss, target_loss, ys, ys_pre, accuracy],
                    feed_dict={u1s: dtest[0:800000, 0], u2s: dtest[0:800000, 1], ys: np.float32(dtest[0:800000, 2])})
                print '----step%5d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (
                i, rmse_v, loss_v, target_loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, t_target_loss_v,
                get_auc(t_y_trues, t_y_pres), t_accuracy_v)
                np.random.shuffle(dtrain)
                np.random.shuffle(dtest)
                continue
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, ys_pre, accuracy],
                                                               feed_dict={u1s: dtrain[0:800000, 0],
                                                                          u2s: dtrain[0:800000, 1],
                                                                          ys: np.float32(dtrain[0:800000, 2])})
        # print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (
        # rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # print 'testing...'
        # rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, ys_pre, accuracy],
        #                                                        feed_dict={u1s: dtest[0:800000, 0],
        #                                                                   u2s: dtest[0:800000, 1],
        #                                                                   ys: np.float32(dtest[0:800000, 2])})
        # print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (
        # rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)

#mf with sigmoid
def exp2(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
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

    target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))
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
        print 'training...'
        rmse_v, loss_v, target_loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
                                                               feed_dict={u1s: dtrain[0:800000, 0],
                                                                          u2s: dtrain[0:800000, 1],
                                                                          ys: np.float32(dtrain[0:800000, 2])})
        t_rmse_v, t_loss_v, t_target_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
                                                                         feed_dict={u1s: dtest[0:800000, 0],
                                                                                    u2s: dtest[0:800000, 1],
                                                                                    ys: np.float32(dtest[0:800000, 2])})
        print '----step%5d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (
        0, rmse_v, loss_v, target_loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, t_target_loss_v, get_auc(t_y_trues, t_y_pres),
        t_accuracy_v)
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                rmse_v, loss_v, target_loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
                                                                       feed_dict={u1s: dtrain[0:800000, 0],
                                                                                  u2s: dtrain[0:800000, 1],
                                                                                  ys: np.float32(dtrain[0:800000, 2])})
                t_rmse_v, t_loss_v, t_target_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run(
                    [rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
                    feed_dict={u1s: dtest[0:800000, 0], u2s: dtest[0:800000, 1], ys: np.float32(dtest[0:800000, 2])})
                print '----step%5d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (
                i, rmse_v, loss_v, target_loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, t_target_loss_v,
                get_auc(t_y_trues, t_y_pres), t_accuracy_v)
                np.random.shuffle(dtrain)
                np.random.shuffle(dtest)
                continue
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
        # rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy],
        #                                                        feed_dict={u1s: dtrain[0:800000, 0],
        #                                                                   u2s: dtrain[0:800000, 1],
        #                                                                   ys: np.float32(dtrain[0:800000, 2])})
        # print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (
        # rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # print 'testing...'
        # rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy],
        #                                                        feed_dict={u1s: dtest[0:800000, 0],
        #                                                                   u2s: dtest[0:800000, 1],
        #                                                                   ys: np.float32(dtest[0:800000, 2])})
        # print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (
        # rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)

#dot > h1 > output
def exp3(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.01, 0.01))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.01, 0.01))

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    weights1 = tf.Variable(tf.random_uniform([params['embedding_size'], params['h1_size']], -0.01, 0.01))
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(dot_e, weights1) + biases1

    weights2 = tf.Variable(tf.random_uniform([params['h1_size'], 1], -0.01, 0.01))
    biases2 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))

    # train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    var_list1 = [p, q]
    var_list2 = [weights1, weights2, biases1, biases2]
    op1 = tf.train.AdamOptimizer(params['learning_rate1']).minimize(loss, var_list = var_list1)
    op2 = tf.train.AdamOptimizer(params['learning_rate2']).minimize(loss, var_list = var_list2)
    train_step = tf.group(op1, op2)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        np.random.shuffle(dtrain)
        data_size = dtrain.shape[0]
        np.random.shuffle(dtrain)
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                np.random.shuffle(dtrain)
                continue
            if i % 1000 == 0:
                tmp, rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([weights2, rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)

#simple pnn
def exp4(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    embedding = tf.Variable(tf.random_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(embedding, u1s)
    e2 = tf.nn.embedding_lookup(embedding, u2s)

    z = tf.concat(1, [e1, e2])
    p = tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)), [-1, params['embedding_size']*params['embedding_size']])

    h0 = tf.concat(1, [z, p])

    weight1 = tf.Variable(tf.random_normal([params['embedding_size']*(params['embedding_size']+2), params['h1_size']], mean=0, stddev=0.01))
    biase1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.nn.relu(tf.matmul(h0, weight1)+biase1)

    weight2 = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0, stddev=0.01))
    biase2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.nn.relu(tf.matmul(h1, weight2)+biase2)

    weight3 = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0, stddev=0.01))
    biase3 = tf.Variable(tf.zeros([params['h3_size']]))
    h3 = tf.nn.relu(tf.matmul(h2, weight3) + biase3)

    weight4 = tf.Variable(tf.random_normal([params['h3_size'], params['h4_size']], mean=0, stddev=0.01))
    biase4 = tf.Variable(tf.zeros([params['h4_size']]))
    h4 = tf.nn.relu(tf.matmul(h3, weight4) + biase4)

    weighto = tf.Variable(tf.random_normal([params['h4_size'], 1], mean=0, stddev=0.01))
    biaseo = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h4, weighto)+biaseo)

    weight_l2 = tf.reduce_sum([tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3), tf.nn.l2_loss(weight4), tf.nn.l2_loss(weighto)])

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
        print 'training...'
        rmse_v, loss_v, target_loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[0:800000, 0], u2s: dtrain[0:800000, 1], ys: np.float32(dtrain[0:800000, 2])})
        t_rmse_v, t_loss_v, t_target_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[0:800000, 0], u2s: dtest[0:800000, 1], ys: np.float32(dtest[0:800000, 2])})
        print '----step%5d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (0, rmse_v, loss_v, target_loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, t_target_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                rmse_v, loss_v, target_loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[0:800000, 0], u2s: dtrain[0:800000, 1], ys: np.float32(dtrain[0:800000, 2])})
                t_rmse_v, t_loss_v, t_target_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[0:800000, 0], u2s: dtest[0:800000, 1], ys: np.float32(dtest[0:800000, 2])})
                print '----step%5d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, target_loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, t_target_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
                np.random.shuffle(dtrain)
                np.random.shuffle(dtest)
                if get_auc(y_trues, y_pres)>0.99:
                    weight1_v, biase1_v, weight2_v, biase2_v, weight3_v, biase3_v, weight4_v, biase4_v, weighto_v, biaseo_v, embedding_v = sess.run([weight1, biase1, weight2, biase2, weight3, biase3, weight4, biase4, weighto, biaseo, embedding], feed_dict={u1s: dtrain[0:0, 0], u2s: dtrain[0:0, 1], ys: np.float32(dtrain[0:0, 2])})
                    save_dict = {'weight1_v': weight1_v, 'biase1_v': biase1_v, 'weight2_v': weight2_v, 'biase2_v': biase2_v, 'weight3_v': weight3_v, 'biase3_v': biase3_v, 'weight4_v': weight4_v, 'biase4_v': biase4_v, 'weighto_v': weighto_v, 'biaseo_v': biaseo_v, 'embedding_v': embedding_v}
                    store_dict(save_dict, params['model_save_path'])
                    break
                continue
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
        # rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy],feed_dict={u1s: dtrain[0:800000, 0], u2s: dtrain[0:800000, 1],ys: np.float32(dtrain[0:800000, 2])})
        # print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # print 'testing...'
        # rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[0:800000, 0], u2s: dtest[0:800000, 1], ys: np.float32(dtest[0:800000, 2])})
        # print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)

def exp4_1(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
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

    # weight1 = tf.Variable(tf.random_normal([params['embedding_size'] * (params['embedding_size'] + 2), params['h1_size']], mean=0, stddev=0.01))
    # biase1 = tf.Variable(tf.zeros([params['h1_size']]))
    # h1 = tf.nn.relu(tf.matmul(h0, weight1) + biase1)
    #
    # weight2 = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0, stddev=0.01))
    # biase2 = tf.Variable(tf.zeros([params['h2_size']]))
    # h2 = tf.nn.relu(tf.matmul(h1, weight2) + biase2)
    #
    # weight3 = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0, stddev=0.01))
    # biase3 = tf.Variable(tf.zeros([params['h3_size']]))
    # h3 = tf.nn.relu(tf.matmul(h2, weight3) + biase3)
    #
    # weight4 = tf.Variable(tf.random_normal([params['h3_size'], params['h4_size']], mean=0, stddev=0.01))
    # biase4 = tf.Variable(tf.zeros([params['h4_size']]))
    # h4 = tf.nn.relu(tf.matmul(h3, weight4) + biase4)
    #
    # weighto = tf.Variable(tf.random_normal([params['h4_size'], 1], mean=0, stddev=0.01))
    # biaseo = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h4, weighto) + biaseo)
    #
    # weight_l2 = tf.reduce_sum(
    #     [tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3), tf.nn.l2_loss(weight4),
    #      tf.nn.l2_loss(weighto)])
    params['h0_size'] = params['embedding_size'] * (params['embedding_size'] + 2)
    w_list = []
    b_list = []
    h_list = [h0]
    for i in range(params['h_num']):
        w_list.append(tf.Variable(tf.random_normal([params['h%d_size' % i], params['h%d_size' % (i+1)]], mean=0, stddev=0.01)))
        b_list.append(tf.Variable(tf.zeros([params['h%d_size' % (i+1)]])))
        h_list.append(tf.nn.relu(tf.matmul(h_list[i], w_list[i]) + b_list[i]))
    weighto = tf.Variable(tf.random_normal([params['h%d_size' % params['h_num']], 1], mean=0, stddev=0.01))
    biaseo = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h_list[params['h_num']], weighto) + biaseo)
    w_list.append(weighto)

    weight_l2 = tf.reduce_sum([tf.nn.l2_loss(w) for w in w_list])

    target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))
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
        print 'training...'
        rmse_v, loss_v, target_loss_v, y_trues, y_pres, accuracy_v = sess.run(
            [rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
            feed_dict={u1s: dtrain[0:800000, 0], u2s: dtrain[0:800000, 1], ys: np.float32(dtrain[0:800000, 2])})
        t_rmse_v, t_loss_v, t_target_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run(
            [rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
            feed_dict={u1s: dtest[0:800000, 0], u2s: dtest[0:800000, 1], ys: np.float32(dtest[0:800000, 2])})
        print '----step%5d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (
        0, rmse_v, loss_v, target_loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, t_target_loss_v,
        get_auc(t_y_trues, t_y_pres), t_accuracy_v)
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                rmse_v, loss_v, target_loss_v, y_trues, y_pres, accuracy_v = sess.run(
                    [rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
                    feed_dict={u1s: dtrain[0:800000, 0], u2s: dtrain[0:800000, 1],
                               ys: np.float32(dtrain[0:800000, 2])})
                t_rmse_v, t_loss_v, t_target_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run(
                    [rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy],
                    feed_dict={u1s: dtest[0:800000, 0], u2s: dtest[0:800000, 1],
                               ys: np.float32(dtest[0:800000, 2])})
                print '----step%5d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (
                i, rmse_v, loss_v, target_loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v,
                t_target_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
                np.random.shuffle(dtrain)
                np.random.shuffle(dtest)
                if get_auc(y_trues, y_pres) > 0.99:
                    weight1_v, biase1_v, weight2_v, biase2_v, weight3_v, biase3_v, weight4_v, biase4_v, weighto_v, biaseo_v, embedding_v = sess.run(
                        [weight1, biase1, weight2, biase2, weight3, biase3, weight4, biase4, weighto, biaseo,
                         embedding],
                        feed_dict={u1s: dtrain[0:0, 0], u2s: dtrain[0:0, 1], ys: np.float32(dtrain[0:0, 2])})
                    save_dict = {'weight1_v': weight1_v, 'biase1_v': biase1_v, 'weight2_v': weight2_v,
                                 'biase2_v': biase2_v, 'weight3_v': weight3_v, 'biase3_v': biase3_v,
                                 'weight4_v': weight4_v, 'biase4_v': biase4_v, 'weighto_v': weighto_v,
                                 'biaseo_v': biaseo_v, 'embedding_v': embedding_v}
                    store_dict(save_dict, params['model_save_path'])
                    break
                continue
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy],feed_dict={u1s: dtrain[0:800000, 0], u2s: dtrain[0:800000, 1],ys: np.float32(dtrain[0:800000, 2])})
            # print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
            # print 'testing...'
            # rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[0:800000, 0], u2s: dtest[0:800000, 1], ys: np.float32(dtest[0:800000, 2])})
            # print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)

#simple pnn with multiple feature, concat all z and p
def exp5(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    # dtrain -= [1, 1, 0]
    # dtest -= [1, 1, 0]

    feature_num = len(params['feature_depth'])
    embedding_list = []
    for i in range(feature_num):
        embedding_list.append(tf.Variable(tf.random_uniform([params['feature_depth'][i], params['embedding_size']], -0.01, 0.01)))

    features = tf.placeholder(tf.int32, shape=[None, feature_num*2])
    ys = tf.placeholder(tf.float32, shape=[None])

    feature_list = tf.unpack(features, axis=1)

    z = tf.concat(1, [tf.nn.embedding_lookup(embedding_list[i/2], feature_list[i]) for i in range(feature_num*2)])
    p = tf.concat(1, [tf.reshape(tf.batch_matmul(tf.expand_dims(tf.nn.embedding_lookup(embedding_list[i], feature_list[2*i]), axis=2), tf.expand_dims(tf.nn.embedding_lookup(embedding_list[i], feature_list[2*i+1]), axis=1)), [-1, params['embedding_size'] * params['embedding_size']]) for i in range(feature_num)])

    h0 = tf.concat(1, [z, p])

    weight1 = tf.Variable(tf.random_normal([feature_num*params['embedding_size']*(params['embedding_size']+2), params['h1_size']], mean=0, stddev=0.01))
    biase1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.nn.relu(tf.matmul(h0, weight1)+biase1)

    weight2 = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0, stddev=0.01))
    biase2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.nn.relu(tf.matmul(h1, weight2)+biase2)

    weight3 = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0, stddev=0.01))
    biase3 = tf.Variable(tf.zeros([params['h3_size']]))
    h3 = tf.nn.relu(tf.matmul(h2, weight3) + biase3)

    weight4 = tf.Variable(tf.random_normal([params['h3_size'], params['h4_size']], mean=0, stddev=0.01))
    biase4 = tf.Variable(tf.zeros([params['h4_size']]))
    h4 = tf.nn.relu(tf.matmul(h3, weight4) + biase4)

    weighto = tf.Variable(tf.random_normal([params['h4_size'], 1], mean=0, stddev=0.01))
    biaseo = tf.zeros([1])
    ys_pre = tf.squeeze(tf.matmul(h4, weighto)+biaseo)

    weight_mss = tf.reduce_mean([tf.reduce_mean(tf.square(weight1)), tf.reduce_mean(tf.square(weight2)), tf.reduce_mean(tf.square(weight3)), tf.reduce_mean(tf.square(weight4)), tf.reduce_mean(tf.square(weighto))])
    biase_mss = tf.reduce_mean([tf.reduce_mean(tf.square(biase1)), tf.reduce_mean(tf.square(biase2)), tf.reduce_mean(tf.square(biase3)), tf.reduce_mean(tf.square(biase4)), tf.reduce_mean(tf.square(biaseo))])

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (weight_mss + biase_mss)

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        data_size = dtrain.shape[0]
        np.random.shuffle(dtrain)
        np.random.shuffle(dtest)
        print 'training...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num])})
        t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num])})
        print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (0, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
                np.random.shuffle(dtrain)
                np.random.shuffle(dtest)
                continue
            feed_dict = {features: dtrain[start:end, 0:2*feature_num], ys: np.float32(dtrain[start:end, 2*feature_num])}
            sess.run(train_step, feed_dict)
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy],feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)

#simple pnn with multiple feature, add all h0(concat(z, p)), train graph feature first
def exp6(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    # dtrain -= [1, 1, 0]
    # dtest -= [1, 1, 0]

    feature_num = len(params['feature_depth'])
    pre_model = load_dict(params['pre_model_save_path'])
    embedding_list = [tf.Variable(pre_model['embedding_v'])]
    for i in range(1, feature_num):
        embedding_list.append(tf.Variable(tf.truncated_normal([params['feature_depth'][i], params['embedding_size']], mean=0, stddev=0.01)))

    features = tf.placeholder(tf.int32, shape=[None, feature_num*2])
    ys = tf.placeholder(tf.float32, shape=[None])

    feature_list = tf.unpack(features, axis=1)

    z_list = [tf.nn.embedding_lookup(embedding_list[i/2], feature_list[i]) for i in range(feature_num*2)]
    p_list = [tf.reshape(tf.batch_matmul(tf.expand_dims(tf.nn.embedding_lookup(embedding_list[i], feature_list[2*i]), axis=2), tf.expand_dims(tf.nn.embedding_lookup(embedding_list[i], feature_list[2*i+1]), axis=1)), [-1, params['embedding_size'] * params['embedding_size']]) for i in range(feature_num)]

    h0_list = []
    for i in range(feature_num):
        h0_list.append(tf.concat(1, [z_list[2*i], z_list[2*i+1], p_list[i]]))
    h0 = tf.reduce_sum(h0_list, axis=0)

    weight1 = tf.Variable(pre_model['weight1_v'])
    biase1 = tf.Variable(pre_model['biase1_v'])
    h1 = tf.nn.relu(tf.matmul(h0, weight1) + biase1)

    # weight2 = tf.Variable(tf.random_normal([feature_num*params['h1_size'], params['h2_size']], mean=0, stddev=0.01))
    weight2 = tf.Variable(pre_model['weight2_v'])
    biase2 = tf.Variable(pre_model['biase2_v'])
    h2 = tf.nn.relu(tf.matmul(h1, weight2)+biase2)

    weight3 = tf.Variable(pre_model['weight3_v'])
    biase3 = tf.Variable(pre_model['biase3_v'])
    h3 = tf.nn.relu(tf.matmul(h2, weight3) + biase3)

    weight4 = tf.Variable(pre_model['weight4_v'])
    biase4 = tf.Variable(pre_model['biase4_v'])
    h4 = tf.nn.relu(tf.matmul(h3, weight4) + biase4)

    weighto = tf.Variable(pre_model['weighto_v'])
    biaseo = tf.Variable(pre_model['biaseo_v'])
    ys_pre = tf.squeeze(tf.matmul(h4, weighto)+biaseo)

    weight_mss = tf.reduce_sum([tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3), tf.nn.l2_loss(weight4), tf.nn.l2_loss(weighto)])
    # biase_mss = tf.reduce_mean([tf.reduce_mean(tf.square(tf.concat(0, biase1_list))), tf.reduce_mean(tf.square(biase2)), tf.reduce_mean(tf.square(biase3)), tf.reduce_mean(tf.square(biase4)), tf.reduce_mean(tf.square(biaseo))])

    target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))
    loss = target_loss + params['beta'] * weight_mss

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        data_size = dtrain.shape[0]
        np.random.shuffle(dtrain)
        np.random.shuffle(dtest)
        print 'training...'
        rmse_v, loss_v, target_loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num])})
        t_rmse_v, t_loss_v, t_target_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num])})
        print '----step%5d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (0, rmse_v, loss_v, target_loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, t_target_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                rmse_v, loss_v, target_loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num])})
                t_rmse_v, t_loss_v, t_target_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num])})
                print '----step%5d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, target_loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, t_target_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
                np.random.shuffle(dtrain)
                np.random.shuffle(dtest)
                continue
            feed_dict = {features: dtrain[start:end, 0:2*feature_num], ys: np.float32(dtrain[start:end, 2*feature_num])}
            sess.run(train_step, feed_dict)
        # rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy],feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num])})
        # print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # print 'testing...'
        # rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num])})
        # print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)

#simple pnn with multiple feature, add all h1, train graph feature first
def exp7(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')

    feature_num = len(params['feature_depth'])
    pre_model = load_dict(params['pre_model_save_path'])
    embedding_list = [tf.Variable(pre_model['embedding_v'])]
    for i in range(1, feature_num):
        embedding_list.append(tf.Variable(tf.truncated_normal([params['feature_depth'][i], params['embedding_size']], mean=0, stddev=4e-3)))
    features = tf.placeholder(tf.int32, shape=[None, feature_num*2])
    ys = tf.placeholder(tf.float32, shape=[None])

    feature_list = tf.unpack(features, axis=1)

    z_list = [tf.nn.embedding_lookup(embedding_list[i/2], feature_list[i]) for i in range(feature_num*2)]
    p_list = [tf.reshape(tf.batch_matmul(tf.expand_dims(tf.nn.embedding_lookup(embedding_list[i], feature_list[2*i]), axis=2), tf.expand_dims(tf.nn.embedding_lookup(embedding_list[i], feature_list[2*i+1]), axis=1)), [-1, params['embedding_size'] * params['embedding_size']]) for i in range(feature_num)]

    h0_list = []
    weight1_list = []
    biase1_list = []
    h1_list = []
    for i in range(feature_num):
        h0_list.append(tf.concat(1, [z_list[2*i], z_list[2*i+1], p_list[i]]))
        if i==0:
            weight1_list.append(tf.Variable(pre_model['weight1_v']))
            biase1_list.append(tf.Variable(pre_model['biase1_v']))
        else:
            weight1_list.append(tf.Variable(tf.truncated_normal([params['embedding_size'] * (params['embedding_size'] + 2), params['h1_size']], mean=0, stddev=4e-3)))
            biase1_list.append(tf.Variable(tf.zeros([params['h1_size']])))
        h1_list.append(tf.nn.relu(tf.matmul(h0_list[i], weight1_list[i])+biase1_list[i]))
    # h1 = tf.concat(1, h1_list)
    h1 = tf.reduce_sum(h1_list, axis=0)

    # weight2 = tf.Variable(tf.random_normal([feature_num*params['h1_size'], params['h2_size']], mean=0, stddev=0.01))
    weight2 = tf.Variable(pre_model['weight2_v'])
    biase2 = tf.Variable(pre_model['biase2_v'])
    h2 = tf.nn.relu(tf.matmul(h1, weight2)+biase2)

    weight3 = tf.Variable(pre_model['weight3_v'])
    biase3 = tf.Variable(pre_model['biase3_v'])
    h3 = tf.nn.relu(tf.matmul(h2, weight3) + biase3)

    weight4 = tf.Variable(pre_model['weight4_v'])
    biase4 = tf.Variable(pre_model['biase4_v'])
    h4 = tf.nn.relu(tf.matmul(h3, weight4) + biase4)

    weighto = tf.Variable(pre_model['weighto_v'])
    biaseo = tf.Variable(pre_model['biaseo_v'])

    ys_pre = tf.squeeze(tf.matmul(h4, weighto)+biaseo)

    weight_mss = tf.reduce_sum([tf.nn.l2_loss(tf.concat(0, weight1_list)), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3), tf.nn.l2_loss(weight4), tf.nn.l2_loss(weighto)])
    # biase_mss = tf.reduce_sum([tf.nn.l2_loss(tf.concat(0, biase1_list)), tf.nn.l2_loss(biase2), tf.nn.l2_loss(biase3), tf.nn.l2_loss(biase4), tf.nn.l2_loss(biaseo)])

    target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))
    loss = target_loss + params['beta'] * weight_mss

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        data_size = dtrain.shape[0]
        np.random.shuffle(dtrain)
        np.random.shuffle(dtest)
        print 'training...'
        rmse_v, loss_v, target_loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num])})
        t_rmse_v, t_loss_v, t_target_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num])})
        print '----step%5d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (0, rmse_v, loss_v, target_loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, t_target_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                rmse_v, loss_v, target_loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num])})
                t_rmse_v, t_loss_v, t_target_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, target_loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num])})
                print '----step%5d: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, target_loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, target_loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, t_target_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
                np.random.shuffle(dtrain)
                np.random.shuffle(dtest)
                continue
            feed_dict = {features: dtrain[start:end, 0:2*feature_num], ys: np.float32(dtrain[start:end, 2*feature_num])}
            sess.run(train_step, feed_dict)
        # rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy],feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num])})
        # print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # print 'testing...'
        # rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num])})
        # print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)

#simple pnn with one feature, plus several network which concat(e, one_hot), with initialization
def exp8(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain_list = np.split(dtrain, [2,4,6,8,10,12,14], axis=1)
    for j in range(params['np_rate']):
        for i in range(len(params['feature_depth'])-1):
            negative_sample = (dtrain_list[i+1] + np.random.randint(low=1, high=params['feature_depth'][i+1], size=(dtrain.shape[0], 2))) % params['feature_depth'][i+1]
            dtrain = np.concatenate((dtrain, negative_sample), axis=1)

    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest_list = np.split(dtest, [2,4,6,8,10,12,14], axis=1)
    for j in range(params['np_rate']):
        for i in range(len(params['feature_depth'])-1):
            negative_sample = (dtest_list[i + 1] + np.random.randint(low=1, high=params['feature_depth'][i+1], size=(dtest.shape[0], 2))) % params['feature_depth'][i+1]
            dtest = np.concatenate((dtest, negative_sample), axis=1)

    # print dtrain[0]
    # print dtest[0]

    feature_num = len(params['feature_depth'])
    pre_model = load_dict(params['pre_model_save_path'])
    embedding = tf.Variable(pre_model['embedding_v'])
    features = tf.placeholder(tf.int32, shape=[None, feature_num*2])
    ys = tf.placeholder(tf.float32, shape=[None])
    n_samples = tf.placeholder(tf.int32, shape=[None, (feature_num-1)*2*params['np_rate']])

    feature_list = tf.unpack(features, axis=1)
    n_sample_list = tf.unpack(n_samples, axis=1)

    e1 = tf.nn.embedding_lookup(embedding, feature_list[0])
    e2 = tf.nn.embedding_lookup(embedding, feature_list[1])
    z = tf.concat(1, [e1, e2])
    p = tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)), [-1, params['embedding_size']*params['embedding_size']])

    h0 = tf.concat(1, [z, p])

    weight1 = tf.Variable(pre_model['weight1_v'])
    biase1 = tf.Variable(pre_model['biase1_v'])
    h1 = tf.nn.relu(tf.matmul(h0, weight1) + biase1)

    weight2 = tf.Variable(pre_model['weight2_v'])
    biase2 = tf.Variable(pre_model['biase2_v'])
    h2 = tf.nn.relu(tf.matmul(h1, weight2) + biase2)

    weight3 = tf.Variable(pre_model['weight3_v'])
    biase3 = tf.Variable(pre_model['biase3_v'])
    h3 = tf.nn.relu(tf.matmul(h2, weight3) + biase3)

    weight4 = tf.Variable(pre_model['weight4_v'])
    biase4 = tf.Variable(pre_model['biase4_v'])
    h4 = tf.nn.relu(tf.matmul(h3, weight4) + biase4)

    weighto = tf.Variable(pre_model['weighto_v'])
    biaseo = tf.Variable(pre_model['biaseo_v'])

    ys_pre = tf.squeeze(tf.matmul(h4, weighto) + biaseo)

    weight_l2 = tf.reduce_sum([tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3), tf.nn.l2_loss(weight4), tf.nn.l2_loss(weighto)])
    biase_l2 = tf.reduce_sum([tf.nn.l2_loss(biase1), tf.nn.l2_loss(biase2), tf.nn.l2_loss(biase3), tf.nn.l2_loss(biase4), tf.nn.l2_loss(biaseo)])

    main_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (weight_l2 + biase_l2)

    ph0_list = []
    pw1_list = []
    pb1_list = []
    ph1_list = []
    pw2_list = []
    pb2_list = []
    ph2_list = []
    pw3_list = []
    pb3_list = []
    ph3_list = []
    pw4_list = []
    pb4_list = []
    ph4_list = []
    pwo_list = []
    pbo_list = []
    py_pre_list = []
    py_list = []
    weight_l2_list = []
    biase_l2_list = []
    ploss_list = []
    ee = tf.concat(0, [e1, e2])
    for i in range(feature_num-1):
        tmp_list = []
        tmp_list.append(tf.concat(1, [ee, tf.concat(0, [tf.one_hot(feature_list[(i+1) * 2], params['feature_depth'][i+1]), tf.one_hot(feature_list[(i+1) * 2 + 1], params['feature_depth'][i+1])])]))
        for j in range(params['np_rate']):
            tmp_list.append(tf.concat(1, [ee, tf.concat(0, [tf.one_hot(n_sample_list[2*j*(feature_num-1)+2*i], params['feature_depth'][i+1]), tf.one_hot(n_sample_list[2*j*(feature_num-1)+2*i+1], params['feature_depth'][i+1])])]))
        ph0_list.append(tf.concat(0, tmp_list))

        pw1_list.append(tf.Variable(tf.truncated_normal([params['embedding_size'] + params['feature_depth'][i+1], params['ph1_size']], mean=0, stddev=0.01)))
        pb1_list.append(tf.Variable(tf.zeros([params['ph1_size']])))
        ph1_list.append(tf.nn.relu(tf.matmul(ph0_list[i], pw1_list[i]) + pb1_list[i]))

        pw2_list.append(tf.Variable(tf.truncated_normal([params['ph1_size'], params['ph2_size']], mean=0, stddev=0.01)))
        pb2_list.append(tf.Variable(tf.zeros([params['ph2_size']])))
        ph2_list.append(tf.nn.relu(tf.matmul(ph1_list[i], pw2_list[i]) + pb2_list[i]))

        pw3_list.append(tf.Variable(tf.truncated_normal([params['ph2_size'], params['ph3_size']], mean=0, stddev=0.01)))
        pb3_list.append(tf.Variable(tf.zeros([params['ph3_size']])))
        ph3_list.append(tf.nn.relu(tf.matmul(ph2_list[i], pw3_list[i]) + pb3_list[i]))

        pw4_list.append(tf.Variable(tf.truncated_normal([params['ph3_size'], params['ph4_size']], mean=0, stddev=0.01)))
        pb4_list.append(tf.Variable(tf.zeros([params['ph4_size']])))
        ph4_list.append(tf.nn.relu(tf.matmul(ph3_list[i], pw4_list[i]) + pb4_list[i]))

        pwo_list.append(tf.Variable(tf.truncated_normal([params['ph4_size'], 1], mean=0, stddev=0.01)))
        pbo_list.append(tf.Variable(tf.zeros([1])))
        py_pre_list.append(tf.squeeze(tf.nn.relu(tf.matmul(ph4_list[i], pwo_list[i]) + pbo_list[i])))

        py_list.append(tf.concat(0, [tf.reshape([tf.ones_like(ys_pre)]*2, [-1]), tf.reshape([tf.zeros_like(ys_pre)]*2*params['np_rate'], [-1])]))

        weight_l2_list.append(tf.reduce_sum([tf.nn.l2_loss(pw1_list[i]), tf.nn.l2_loss(pw2_list[i]), tf.nn.l2_loss(pw3_list[i]), tf.nn.l2_loss(pw4_list[i]), tf.nn.l2_loss(pwo_list[i])]))
        biase_l2_list.append(tf.reduce_sum([tf.nn.l2_loss(pb1_list[i]), tf.nn.l2_loss(pb2_list[i]), tf.nn.l2_loss(pb3_list[i]), tf.nn.l2_loss(pb4_list[i]), tf.nn.l2_loss(pbo_list[i])]))

        ploss_list.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(py_pre_list[i], py_list[i])) + params['pbeta'] * (weight_l2_list[i] + biase_l2_list[i]))

    plu_loss = tf.reduce_sum(ploss_list)
    loss = main_loss + params['lamda'] * plu_loss

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        data_size = dtrain.shape[0]
        np.random.shuffle(dtrain)
        np.random.shuffle(dtest)
        print 'training...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
        t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
        print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (0, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
                np.random.shuffle(dtrain)
                np.random.shuffle(dtest)
                continue
            feed_dict = {features: dtrain[start:end, 0:2*feature_num], ys: np.float32(dtrain[start:end, 2*feature_num]), n_samples: dtrain[start:end, 2*feature_num+1:]}
            sess.run(train_step, feed_dict)
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy],feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)

#simple pnn with one feature, plus several network, without initialization
def exp9(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain_list = np.split(dtrain, [2,4,6,8,10,12,14], axis=1)
    for j in range(params['np_rate']):
        for i in range(len(params['feature_depth'])-1):
            negative_sample = (dtrain_list[i+1] + np.random.randint(low=1, high=params['feature_depth'][i+1], size=(dtrain.shape[0], 2))) % params['feature_depth'][i+1]
            dtrain = np.concatenate((dtrain, negative_sample), axis=1)

    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest_list = np.split(dtest, [2,4,6,8,10,12,14], axis=1)
    for j in range(params['np_rate']):
        for i in range(len(params['feature_depth'])-1):
            negative_sample = (dtest_list[i + 1] + np.random.randint(low=1, high=params['feature_depth'][i+1], size=(dtest.shape[0], 2))) % params['feature_depth'][i+1]
            dtest = np.concatenate((dtest, negative_sample), axis=1)

    # print dtrain[0]
    # print dtest[0]

    feature_num = len(params['feature_depth'])
    embedding = tf.Variable(tf.random_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))
    features = tf.placeholder(tf.int32, shape=[None, feature_num*2])
    ys = tf.placeholder(tf.float32, shape=[None])
    n_samples = tf.placeholder(tf.int32, shape=[None, (feature_num-1)*2*params['np_rate']])

    feature_list = tf.unpack(features, axis=1)
    n_sample_list = tf.unpack(n_samples, axis=1)

    e1 = tf.nn.embedding_lookup(embedding, feature_list[0])
    e2 = tf.nn.embedding_lookup(embedding, feature_list[1])
    z = tf.concat(1, [e1, e2])
    p = tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)), [-1, params['embedding_size']*params['embedding_size']])

    h0 = tf.concat(1, [z, p])

    weight1 = tf.Variable(
        tf.random_normal([params['embedding_size'] * (params['embedding_size'] + 2), params['h1_size']], mean=0,
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

    weight_l2 = tf.reduce_sum([tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3), tf.nn.l2_loss(weight4), tf.nn.l2_loss(weighto)])
    biase_l2 = tf.reduce_sum([tf.nn.l2_loss(biase1), tf.nn.l2_loss(biase2), tf.nn.l2_loss(biase3), tf.nn.l2_loss(biase4), tf.nn.l2_loss(biaseo)])

    main_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (weight_l2 + biase_l2)

    ph0_list = []
    pw1_list = []
    pb1_list = []
    ph1_list = []
    pw2_list = []
    pb2_list = []
    ph2_list = []
    pw3_list = []
    pb3_list = []
    ph3_list = []
    pw4_list = []
    pb4_list = []
    ph4_list = []
    pwo_list = []
    pbo_list = []
    py_pre_list = []
    py_list = []
    weight_l2_list = []
    biase_l2_list = []
    ploss_list = []
    ee = tf.concat(0, [e1, e2])
    for i in range(feature_num-1):
        tmp_list = []
        tmp_list.append(tf.concat(1, [ee, tf.concat(0, [tf.one_hot(feature_list[(i+1) * 2], params['feature_depth'][i+1]), tf.one_hot(feature_list[(i+1) * 2 + 1], params['feature_depth'][i+1])])]))
        for j in range(params['np_rate']):
            tmp_list.append(tf.concat(1, [ee, tf.concat(0, [tf.one_hot(n_sample_list[2*j*(feature_num-1)+2*i], params['feature_depth'][i+1]), tf.one_hot(n_sample_list[2*j*(feature_num-1)+2*i+1], params['feature_depth'][i+1])])]))
        ph0_list.append(tf.concat(0, tmp_list))

        pw1_list.append(tf.Variable(tf.truncated_normal([params['embedding_size'] + params['feature_depth'][i+1], params['ph1_size']], mean=0, stddev=0.01)))
        pb1_list.append(tf.Variable(tf.zeros([params['ph1_size']])))
        ph1_list.append(tf.nn.relu(tf.matmul(ph0_list[i], pw1_list[i]) + pb1_list[i]))

        pw2_list.append(tf.Variable(tf.truncated_normal([params['ph1_size'], params['ph2_size']], mean=0, stddev=0.01)))
        pb2_list.append(tf.Variable(tf.zeros([params['ph2_size']])))
        ph2_list.append(tf.nn.relu(tf.matmul(ph1_list[i], pw2_list[i]) + pb2_list[i]))

        pw3_list.append(tf.Variable(tf.truncated_normal([params['ph2_size'], params['ph3_size']], mean=0, stddev=0.01)))
        pb3_list.append(tf.Variable(tf.zeros([params['ph3_size']])))
        ph3_list.append(tf.nn.relu(tf.matmul(ph2_list[i], pw3_list[i]) + pb3_list[i]))

        pw4_list.append(tf.Variable(tf.truncated_normal([params['ph3_size'], params['ph4_size']], mean=0, stddev=0.01)))
        pb4_list.append(tf.Variable(tf.zeros([params['ph4_size']])))
        ph4_list.append(tf.nn.relu(tf.matmul(ph3_list[i], pw4_list[i]) + pb4_list[i]))

        pwo_list.append(tf.Variable(tf.truncated_normal([params['ph4_size'], 1], mean=0, stddev=0.01)))
        pbo_list.append(tf.Variable(tf.zeros([1])))
        py_pre_list.append(tf.squeeze(tf.nn.relu(tf.matmul(ph4_list[i], pwo_list[i]) + pbo_list[i])))

        py_list.append(tf.concat(0, [tf.reshape([tf.ones_like(ys_pre)]*2, [-1]), tf.reshape([tf.zeros_like(ys_pre)]*2*params['np_rate'], [-1])]))

        weight_l2_list.append(tf.reduce_sum([tf.nn.l2_loss(pw1_list[i]), tf.nn.l2_loss(pw2_list[i]), tf.nn.l2_loss(pw3_list[i]), tf.nn.l2_loss(pw4_list[i]), tf.nn.l2_loss(pwo_list[i])]))
        biase_l2_list.append(tf.reduce_sum([tf.nn.l2_loss(pb1_list[i]), tf.nn.l2_loss(pb2_list[i]), tf.nn.l2_loss(pb3_list[i]), tf.nn.l2_loss(pb4_list[i]), tf.nn.l2_loss(pbo_list[i])]))

        ploss_list.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(py_pre_list[i], py_list[i])) + params['pbeta'] * (weight_l2_list[i] + biase_l2_list[i]))

    plu_loss = tf.reduce_sum(ploss_list)
    loss = main_loss + params['lamda'] * plu_loss

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        data_size = dtrain.shape[0]
        np.random.shuffle(dtrain)
        np.random.shuffle(dtest)
        print 'training...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
        t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
        print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (0, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
                np.random.shuffle(dtrain)
                np.random.shuffle(dtest)
                continue
            feed_dict = {features: dtrain[start:end, 0:2*feature_num], ys: np.float32(dtrain[start:end, 2*feature_num]), n_samples: dtrain[start:end, 2*feature_num+1:]}
            sess.run(train_step, feed_dict)
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy],feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)

#simple pnn with one feature, plus several network with less hidden layer, with initialization
def exp10(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain_list = np.split(dtrain, [2,4,6,8,10,12,14], axis=1)
    for j in range(params['np_rate']):
        for i in range(len(params['feature_depth'])-1):
            negative_sample = (dtrain_list[i+1] + np.random.randint(low=1, high=params['feature_depth'][i+1], size=(dtrain.shape[0], 2))) % params['feature_depth'][i+1]
            dtrain = np.concatenate((dtrain, negative_sample), axis=1)

    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest_list = np.split(dtest, [2,4,6,8,10,12,14], axis=1)
    for j in range(params['np_rate']):
        for i in range(len(params['feature_depth'])-1):
            negative_sample = (dtest_list[i + 1] + np.random.randint(low=1, high=params['feature_depth'][i+1], size=(dtest.shape[0], 2))) % params['feature_depth'][i+1]
            dtest = np.concatenate((dtest, negative_sample), axis=1)

    # print dtrain[0]
    # print dtest[0]

    feature_num = len(params['feature_depth'])
    pre_model = load_dict(params['pre_model_save_path'])
    embedding = tf.Variable(pre_model['embedding_v'])
    features = tf.placeholder(tf.int32, shape=[None, feature_num*2])
    ys = tf.placeholder(tf.float32, shape=[None])
    n_samples = tf.placeholder(tf.int32, shape=[None, (feature_num-1)*2*params['np_rate']])

    feature_list = tf.unpack(features, axis=1)
    n_sample_list = tf.unpack(n_samples, axis=1)

    e1 = tf.nn.embedding_lookup(embedding, feature_list[0])
    e2 = tf.nn.embedding_lookup(embedding, feature_list[1])
    z = tf.concat(1, [e1, e2])
    p = tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)), [-1, params['embedding_size']*params['embedding_size']])

    h0 = tf.concat(1, [z, p])

    weight1 = tf.Variable(pre_model['weight1_v'])
    biase1 = tf.Variable(pre_model['biase1_v'])
    h1 = tf.nn.relu(tf.matmul(h0, weight1) + biase1)

    weight2 = tf.Variable(pre_model['weight2_v'])
    biase2 = tf.Variable(pre_model['biase2_v'])
    h2 = tf.nn.relu(tf.matmul(h1, weight2) + biase2)

    weight3 = tf.Variable(pre_model['weight3_v'])
    biase3 = tf.Variable(pre_model['biase3_v'])
    h3 = tf.nn.relu(tf.matmul(h2, weight3) + biase3)

    weight4 = tf.Variable(pre_model['weight4_v'])
    biase4 = tf.Variable(pre_model['biase4_v'])
    h4 = tf.nn.relu(tf.matmul(h3, weight4) + biase4)

    weighto = tf.Variable(pre_model['weighto_v'])
    biaseo = tf.Variable(pre_model['biaseo_v'])

    ys_pre = tf.squeeze(tf.matmul(h4, weighto) + biaseo)

    weight_l2 = tf.reduce_sum([tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3), tf.nn.l2_loss(weight4), tf.nn.l2_loss(weighto)])
    biase_l2 = tf.reduce_sum([tf.nn.l2_loss(biase1), tf.nn.l2_loss(biase2), tf.nn.l2_loss(biase3), tf.nn.l2_loss(biase4), tf.nn.l2_loss(biaseo)])

    main_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (weight_l2 + biase_l2)

    ph0_list = []
    pw1_list = []
    pb1_list = []
    ph1_list = []
    pw2_list = []
    pb2_list = []
    ph2_list = []
    pw3_list = []
    pb3_list = []
    ph3_list = []
    pw4_list = []
    pb4_list = []
    ph4_list = []
    pwo_list = []
    pbo_list = []
    py_pre_list = []
    py_list = []
    weight_l2_list = []
    biase_l2_list = []
    ploss_list = []
    ee = tf.concat(0, [e1, e2])
    for i in range(feature_num-1):
        tmp_list = []
        tmp_list.append(tf.concat(1, [ee, tf.concat(0, [tf.one_hot(feature_list[(i+1) * 2], params['feature_depth'][i+1]), tf.one_hot(feature_list[(i+1) * 2 + 1], params['feature_depth'][i+1])])]))
        for j in range(params['np_rate']):
            tmp_list.append(tf.concat(1, [ee, tf.concat(0, [tf.one_hot(n_sample_list[2*j*(feature_num-1)+2*i], params['feature_depth'][i+1]), tf.one_hot(n_sample_list[2*j*(feature_num-1)+2*i+1], params['feature_depth'][i+1])])]))
        ph0_list.append(tf.concat(0, tmp_list))

        pw1_list.append(tf.Variable(tf.truncated_normal([params['embedding_size'] + params['feature_depth'][i+1], params['ph1_size']], mean=0, stddev=0.01)))
        pb1_list.append(tf.Variable(tf.zeros([params['ph1_size']])))
        ph1_list.append(tf.nn.relu(tf.matmul(ph0_list[i], pw1_list[i]) + pb1_list[i]))

        pw2_list.append(tf.Variable(tf.truncated_normal([params['ph1_size'], params['ph2_size']], mean=0, stddev=0.01)))
        pb2_list.append(tf.Variable(tf.zeros([params['ph2_size']])))
        ph2_list.append(tf.nn.relu(tf.matmul(ph1_list[i], pw2_list[i]) + pb2_list[i]))

        # pw3_list.append(tf.Variable(tf.truncated_normal([params['ph2_size'], params['ph3_size']], mean=0, stddev=0.01)))
        # pb3_list.append(tf.Variable(tf.zeros([params['ph3_size']])))
        # ph3_list.append(tf.nn.relu(tf.matmul(ph2_list[i], pw3_list[i]) + pb3_list[i]))
        #
        # pw4_list.append(tf.Variable(tf.truncated_normal([params['ph3_size'], params['ph4_size']], mean=0, stddev=0.01)))
        # pb4_list.append(tf.Variable(tf.zeros([params['ph4_size']])))
        # ph4_list.append(tf.nn.relu(tf.matmul(ph3_list[i], pw4_list[i]) + pb4_list[i]))

        pwo_list.append(tf.Variable(tf.truncated_normal([params['ph2_size'], 1], mean=0, stddev=0.01)))
        pbo_list.append(tf.Variable(tf.zeros([1])))
        py_pre_list.append(tf.squeeze(tf.nn.relu(tf.matmul(ph2_list[i], pwo_list[i]) + pbo_list[i])))

        py_list.append(tf.concat(0, [tf.reshape([tf.ones_like(ys_pre)]*2, [-1]), tf.reshape([tf.zeros_like(ys_pre)]*2*params['np_rate'], [-1])]))

        # weight_l2_list.append(tf.reduce_sum([tf.nn.l2_loss(pw1_list[i]), tf.nn.l2_loss(pw2_list[i]), tf.nn.l2_loss(pw3_list[i]), tf.nn.l2_loss(pw4_list[i]), tf.nn.l2_loss(pwo_list[i])]))
        # biase_l2_list.append(tf.reduce_sum([tf.nn.l2_loss(pb1_list[i]), tf.nn.l2_loss(pb2_list[i]), tf.nn.l2_loss(pb3_list[i]), tf.nn.l2_loss(pb4_list[i]), tf.nn.l2_loss(pbo_list[i])]))
        weight_l2_list.append(tf.reduce_sum([tf.nn.l2_loss(pw1_list[i]), tf.nn.l2_loss(pw2_list[i]), tf.nn.l2_loss(pwo_list[i])]))
        biase_l2_list.append(tf.reduce_sum([tf.nn.l2_loss(pb1_list[i]), tf.nn.l2_loss(pb2_list[i]), tf.nn.l2_loss(pbo_list[i])]))

        ploss_list.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(py_pre_list[i], py_list[i])) + params['pbeta'] * (weight_l2_list[i] + biase_l2_list[i]))

    plu_loss = tf.reduce_sum(ploss_list)
    loss = main_loss + params['lamda'] * plu_loss

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        data_size = dtrain.shape[0]
        np.random.shuffle(dtrain)
        np.random.shuffle(dtest)
        print 'training...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
        t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
        print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (0, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
                np.random.shuffle(dtrain)
                np.random.shuffle(dtest)
                continue
            feed_dict = {features: dtrain[start:end, 0:2*feature_num], ys: np.float32(dtrain[start:end, 2*feature_num]), n_samples: dtrain[start:end, 2*feature_num+1:]}
            sess.run(train_step, feed_dict)
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy],feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)

#simple pnn with one feature, plus several network with less hidden layer, without initialization, seperated loss, plus network concat
def exp11(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain_list = np.split(dtrain, [2,4,6,8,10,12,14], axis=1)
    for j in range(params['np_rate']):
        for i in range(len(params['feature_depth'])-1):
            negative_sample = (dtrain_list[i+1] + np.random.randint(low=1, high=params['feature_depth'][i+1], size=(dtrain.shape[0], 2))) % params['feature_depth'][i+1]
            dtrain = np.concatenate((dtrain, negative_sample), axis=1)

    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest_list = np.split(dtest, [2,4,6,8,10,12,14], axis=1)
    for j in range(params['np_rate']):
        for i in range(len(params['feature_depth'])-1):
            negative_sample = (dtest_list[i + 1] + np.random.randint(low=1, high=params['feature_depth'][i+1], size=(dtest.shape[0], 2))) % params['feature_depth'][i+1]
            dtest = np.concatenate((dtest, negative_sample), axis=1)

    # print dtrain[0]
    # print dtest[0]

    feature_num = len(params['feature_depth'])
    embedding = tf.Variable(tf.random_normal([params['node_num'], params['embedding_size']], mean=0, stddev=0.01))
    features = tf.placeholder(tf.int32, shape=[None, feature_num*2])
    ys = tf.placeholder(tf.float32, shape=[None])
    n_samples = tf.placeholder(tf.int32, shape=[None, (feature_num-1)*2*params['np_rate']])

    feature_list = tf.unpack(features, axis=1)
    n_sample_list = tf.unpack(n_samples, axis=1)

    e1 = tf.nn.embedding_lookup(embedding, feature_list[0])
    e2 = tf.nn.embedding_lookup(embedding, feature_list[1])
    z = tf.concat(1, [e1, e2])
    p = tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)), [-1, params['embedding_size']*params['embedding_size']])

    h0 = tf.concat(1, [z, p])

    weight1 = tf.Variable(
        tf.random_normal([params['embedding_size'] * (params['embedding_size'] + 2), params['h1_size']], mean=0,
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

    weight_l2 = tf.reduce_sum([tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3), tf.nn.l2_loss(weight4), tf.nn.l2_loss(weighto)])
    biase_l2 = tf.reduce_sum([tf.nn.l2_loss(biase1), tf.nn.l2_loss(biase2), tf.nn.l2_loss(biase3), tf.nn.l2_loss(biase4), tf.nn.l2_loss(biaseo)])

    main_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (weight_l2 + biase_l2)

    ph0_list = []
    pw1_list = []
    pb1_list = []
    ph1_list = []
    pw2_list = []
    pb2_list = []
    ph2_list = []
    pw3_list = []
    pb3_list = []
    ph3_list = []
    pw4_list = []
    pb4_list = []
    ph4_list = []
    pwo_list = []
    pbo_list = []
    py_pre_list = []
    py_list = []
    weight_l2_list = []
    biase_l2_list = []
    ploss_list = []
    ee = tf.concat(0, [e1, e2])
    for i in range(feature_num-1):
        tmp_list = []
        tmp_list.append(tf.concat(1, [ee, tf.concat(0, [tf.one_hot(feature_list[(i+1) * 2], params['feature_depth'][i+1]), tf.one_hot(feature_list[(i+1) * 2 + 1], params['feature_depth'][i+1])])]))
        for j in range(params['np_rate']):
            tmp_list.append(tf.concat(1, [ee, tf.concat(0, [tf.one_hot(n_sample_list[2*j*(feature_num-1)+2*i], params['feature_depth'][i+1]), tf.one_hot(n_sample_list[2*j*(feature_num-1)+2*i+1], params['feature_depth'][i+1])])]))
        ph0_list.append(tf.concat(0, tmp_list))

        pw1_list.append(tf.Variable(tf.truncated_normal([params['embedding_size'] + params['feature_depth'][i+1], params['ph1_size']], mean=0, stddev=0.01)))
        pb1_list.append(tf.Variable(tf.zeros([params['ph1_size']])))
        ph1_list.append(tf.nn.relu(tf.matmul(ph0_list[i], pw1_list[i]) + pb1_list[i]))

        pw2_list.append(tf.Variable(tf.truncated_normal([params['ph1_size'], params['ph2_size']], mean=0, stddev=0.01)))
        pb2_list.append(tf.Variable(tf.zeros([params['ph2_size']])))
        ph2_list.append(tf.nn.relu(tf.matmul(ph1_list[i], pw2_list[i]) + pb2_list[i]))

        # pw3_list.append(tf.Variable(tf.truncated_normal([params['ph2_size'], params['ph3_size']], mean=0, stddev=0.01)))
        # pb3_list.append(tf.Variable(tf.zeros([params['ph3_size']])))
        # ph3_list.append(tf.nn.relu(tf.matmul(ph2_list[i], pw3_list[i]) + pb3_list[i]))
        #
        # pw4_list.append(tf.Variable(tf.truncated_normal([params['ph3_size'], params['ph4_size']], mean=0, stddev=0.01)))
        # pb4_list.append(tf.Variable(tf.zeros([params['ph4_size']])))
        # ph4_list.append(tf.nn.relu(tf.matmul(ph3_list[i], pw4_list[i]) + pb4_list[i]))

        pwo_list.append(tf.Variable(tf.truncated_normal([params['ph2_size'], 1], mean=0, stddev=0.01)))
        pbo_list.append(tf.Variable(tf.zeros([1])))
        py_pre_list.append(tf.squeeze(tf.nn.relu(tf.matmul(ph2_list[i], pwo_list[i]) + pbo_list[i])))

        py_list.append(tf.concat(0, [tf.reshape([tf.ones_like(ys_pre)]*2, [-1]), tf.reshape([tf.zeros_like(ys_pre)]*2*params['np_rate'], [-1])]))

        # weight_l2_list.append(tf.reduce_sum([tf.nn.l2_loss(pw1_list[i]), tf.nn.l2_loss(pw2_list[i]), tf.nn.l2_loss(pw3_list[i]), tf.nn.l2_loss(pw4_list[i]), tf.nn.l2_loss(pwo_list[i])]))
        # biase_l2_list.append(tf.reduce_sum([tf.nn.l2_loss(pb1_list[i]), tf.nn.l2_loss(pb2_list[i]), tf.nn.l2_loss(pb3_list[i]), tf.nn.l2_loss(pb4_list[i]), tf.nn.l2_loss(pbo_list[i])]))
        weight_l2_list.append(tf.reduce_sum([tf.nn.l2_loss(pw1_list[i]), tf.nn.l2_loss(pw2_list[i]), tf.nn.l2_loss(pwo_list[i])]))
        biase_l2_list.append(tf.reduce_sum([tf.nn.l2_loss(pb1_list[i]), tf.nn.l2_loss(pb2_list[i]), tf.nn.l2_loss(pbo_list[i])]))

        ploss_list.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(py_pre_list[i], py_list[i])) + params['pbeta'] * (weight_l2_list[i] + biase_l2_list[i]))

    plu_loss = tf.reduce_sum(ploss_list)
    # loss = main_loss + params['lamda'] * plu_loss
    loss = main_loss

    train_step1 = tf.train.AdamOptimizer(params['learning_rate']).minimize(main_loss)
    train_step2 = tf.train.AdamOptimizer(params['learning_rate']).minimize(plu_loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        data_size = dtrain.shape[0]
        np.random.shuffle(dtrain)
        np.random.shuffle(dtest)
        print 'training...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
        t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
        print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (0, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
                np.random.shuffle(dtrain)
                np.random.shuffle(dtest)
                continue
            feed_dict = {features: dtrain[start:end, 0:2*feature_num], ys: np.float32(dtrain[start:end, 2*feature_num]), n_samples: dtrain[start:end, 2*feature_num+1:]}
            sess.run([train_step1,train_step2], feed_dict)
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy],feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)

#simple pnn with one feature, plus several network which concat(e, e), with initialization
def exp12(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain_list = np.split(dtrain, [2,4,6,8,10,12,14], axis=1)
    for j in range(params['np_rate']):
        for i in range(len(params['feature_depth'])-1):
            negative_sample = (dtrain_list[i+1] + np.random.randint(low=1, high=params['feature_depth'][i+1], size=(dtrain.shape[0], 2))) % params['feature_depth'][i+1]
            dtrain = np.concatenate((dtrain, negative_sample), axis=1)

    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest_list = np.split(dtest, [2,4,6,8,10,12,14], axis=1)
    for j in range(params['np_rate']):
        for i in range(len(params['feature_depth'])-1):
            negative_sample = (dtest_list[i + 1] + np.random.randint(low=1, high=params['feature_depth'][i+1], size=(dtest.shape[0], 2))) % params['feature_depth'][i+1]
            dtest = np.concatenate((dtest, negative_sample), axis=1)

    # print dtrain[0]
    # print dtest[0]

    feature_num = len(params['feature_depth'])
    pre_model = load_dict(params['pre_model_save_path'])
    embedding_list = [tf.Variable(pre_model['embedding_v'])]
    features = tf.placeholder(tf.int32, shape=[None, feature_num*2])
    ys = tf.placeholder(tf.float32, shape=[None])
    n_samples = tf.placeholder(tf.int32, shape=[None, (feature_num-1)*2*params['np_rate']])

    feature_list = tf.unpack(features, axis=1)
    n_sample_list = tf.unpack(n_samples, axis=1)

    e1 = tf.nn.embedding_lookup(embedding_list[0], feature_list[0])
    e2 = tf.nn.embedding_lookup(embedding_list[0], feature_list[1])
    z = tf.concat(1, [e1, e2])
    p = tf.reshape(tf.batch_matmul(tf.expand_dims(e1, 2), tf.expand_dims(e2, 1)), [-1, params['embedding_size']*params['embedding_size']])

    h0 = tf.concat(1, [z, p])

    weight1 = tf.Variable(pre_model['weight1_v'])
    biase1 = tf.Variable(pre_model['biase1_v'])
    h1 = tf.nn.relu(tf.matmul(h0, weight1) + biase1)

    weight2 = tf.Variable(pre_model['weight2_v'])
    biase2 = tf.Variable(pre_model['biase2_v'])
    h2 = tf.nn.relu(tf.matmul(h1, weight2) + biase2)

    weight3 = tf.Variable(pre_model['weight3_v'])
    biase3 = tf.Variable(pre_model['biase3_v'])
    h3 = tf.nn.relu(tf.matmul(h2, weight3) + biase3)

    weight4 = tf.Variable(pre_model['weight4_v'])
    biase4 = tf.Variable(pre_model['biase4_v'])
    h4 = tf.nn.relu(tf.matmul(h3, weight4) + biase4)

    weighto = tf.Variable(pre_model['weighto_v'])
    biaseo = tf.Variable(pre_model['biaseo_v'])

    ys_pre = tf.squeeze(tf.matmul(h4, weighto) + biaseo)

    weight_l2 = tf.reduce_sum([tf.nn.l2_loss(weight1), tf.nn.l2_loss(weight2), tf.nn.l2_loss(weight3), tf.nn.l2_loss(weight4), tf.nn.l2_loss(weighto)])
    biase_l2 = tf.reduce_sum([tf.nn.l2_loss(biase1), tf.nn.l2_loss(biase2), tf.nn.l2_loss(biase3), tf.nn.l2_loss(biase4), tf.nn.l2_loss(biaseo)])

    main_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (weight_l2 + biase_l2)

    ph0_list = []
    pw1_list = []
    pb1_list = []
    ph1_list = []
    pw2_list = []
    pb2_list = []
    ph2_list = []
    pw3_list = []
    pb3_list = []
    ph3_list = []
    pw4_list = []
    pb4_list = []
    ph4_list = []
    pwo_list = []
    pbo_list = []
    py_pre_list = []
    py_list = []
    weight_l2_list = []
    biase_l2_list = []
    ploss_list = []
    ee = tf.concat(0, [e1, e2])
    for i in range(feature_num-1):
        tmp_list = []
        embedding_list.append(tf.Variable(tf.truncated_normal([params['feature_depth'][i+1], params['embedding_size']], mean=0, stddev=0.01)))
        tmp_list.append(tf.concat(1, [ee, tf.concat(0, [tf.nn.embedding_lookup(embedding_list[i+1], feature_list[2*(i+1)]), tf.nn.embedding_lookup(embedding_list[i+1], feature_list[2*(i+1)+1])])]))
        tf.nn.embedding_lookup(embedding_list[i+1], n_sample_list[2*j*(feature_num-1)])
        for j in range(params['np_rate']):
            tmp_list.append(tf.concat(1, [ee, tf.concat(0, [tf.nn.embedding_lookup(embedding_list[i+1], n_sample_list[2*j*(feature_num-1)+2*i]), tf.nn.embedding_lookup(embedding_list[i+1], n_sample_list[2*j*(feature_num-1)+2*i+1])])]))
        ph0_list.append(tf.concat(0, tmp_list))

        pw1_list.append(tf.Variable(tf.truncated_normal([params['embedding_size'] + params['feature_depth'][i+1], params['ph1_size']], mean=0, stddev=0.01)))
        pb1_list.append(tf.Variable(tf.zeros([params['ph1_size']])))
        ph1_list.append(tf.nn.relu(tf.matmul(ph0_list[i], pw1_list[i]) + pb1_list[i]))

        pw2_list.append(tf.Variable(tf.truncated_normal([params['ph1_size'], params['ph2_size']], mean=0, stddev=0.01)))
        pb2_list.append(tf.Variable(tf.zeros([params['ph2_size']])))
        ph2_list.append(tf.nn.relu(tf.matmul(ph1_list[i], pw2_list[i]) + pb2_list[i]))

        pw3_list.append(tf.Variable(tf.truncated_normal([params['ph2_size'], params['ph3_size']], mean=0, stddev=0.01)))
        pb3_list.append(tf.Variable(tf.zeros([params['ph3_size']])))
        ph3_list.append(tf.nn.relu(tf.matmul(ph2_list[i], pw3_list[i]) + pb3_list[i]))

        pw4_list.append(tf.Variable(tf.truncated_normal([params['ph3_size'], params['ph4_size']], mean=0, stddev=0.01)))
        pb4_list.append(tf.Variable(tf.zeros([params['ph4_size']])))
        ph4_list.append(tf.nn.relu(tf.matmul(ph3_list[i], pw4_list[i]) + pb4_list[i]))

        pwo_list.append(tf.Variable(tf.truncated_normal([params['ph4_size'], 1], mean=0, stddev=0.01)))
        pbo_list.append(tf.Variable(tf.zeros([1])))
        py_pre_list.append(tf.squeeze(tf.nn.relu(tf.matmul(ph4_list[i], pwo_list[i]) + pbo_list[i])))

        py_list.append(tf.concat(0, [tf.reshape([tf.ones_like(ys_pre)]*2, [-1]), tf.reshape([tf.zeros_like(ys_pre)]*2*params['np_rate'], [-1])]))

        weight_l2_list.append(tf.reduce_sum([tf.nn.l2_loss(pw1_list[i]), tf.nn.l2_loss(pw2_list[i]), tf.nn.l2_loss(pw3_list[i]), tf.nn.l2_loss(pw4_list[i]), tf.nn.l2_loss(pwo_list[i])]))
        biase_l2_list.append(tf.reduce_sum([tf.nn.l2_loss(pb1_list[i]), tf.nn.l2_loss(pb2_list[i]), tf.nn.l2_loss(pb3_list[i]), tf.nn.l2_loss(pb4_list[i]), tf.nn.l2_loss(pbo_list[i])]))

        ploss_list.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(py_pre_list[i], py_list[i])) + params['pbeta'] * (weight_l2_list[i] + biase_l2_list[i]))

    plu_loss = tf.reduce_sum(ploss_list)
    loss = main_loss + params['lamda'] * plu_loss

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        data_size = dtrain.shape[0]
        np.random.shuffle(dtrain)
        np.random.shuffle(dtest)
        print 'training...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
        t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
        print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (0, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
                np.random.shuffle(dtrain)
                np.random.shuffle(dtest)
                continue
            feed_dict = {features: dtrain[start:end, 0:2*feature_num], ys: np.float32(dtrain[start:end, 2*feature_num]), n_samples: dtrain[start:end, 2*feature_num+1:]}
            sess.run(train_step, feed_dict)
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy],feed_dict={features: dtrain[0:800000, 0:2*feature_num], ys: np.float32(dtrain[0:800000, 2*feature_num]), n_samples: dtrain[0:800000, 2*feature_num+1:]})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={features: dtest[0:800000, 0:2*feature_num], ys: np.float32(dtest[0:800000, 2*feature_num]), n_samples: dtest[0:800000, 2*feature_num+1:]})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)

#only one embedding for one node, train two pnn, one for link prediction, one for tow-lop link prediction, train alternately
def exp13(params):
    pre_model = load_dict(params['pre_model_save_path'])
    embedding = tf.Variable(pre_model['embedding_v'])

    dtrain_a = np.loadtxt(params['dtrain_a_file_path'], delimiter=',')
    dtrain_a -= [1, 1, 0]
    dtest_a = np.loadtxt(params['dtest_a_file_path'], delimiter=',')
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

    weight_l2_a = tf.reduce_sum([tf.nn.l2_loss(weight1_a), tf.nn.l2_loss(weight2_a), tf.nn.l2_loss(weight3_a), tf.nn.l2_loss(weight4_a), tf.nn.l2_loss(weighto_a)])

    target_loss_a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre_a, ys_a))
    loss_a = target_loss_a + params['beta'] * weight_l2_a

    train_step_a = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss_a)

    dtrain_b = np.loadtxt(params['dtrain_b_file_path'], delimiter=',')
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

    weight1_b = tf.Variable(tf.random_normal([params['embedding_size'] * (params['embedding_size'] + 2), params['h1_size']], mean=0, stddev=0.01))
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

    # rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))
    #
    # accuracy = tf.reduce_mean(
    #     tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print 'starting...'

        data_size_a = dtrain_a.shape[0]
        np.random.shuffle(dtrain_a)
        np.random.shuffle(dtest_a)
        loss_a_v, target_loss_a_v, ys_a_v, ys_pre_a_v = sess.run([loss_a, target_loss_a, ys_a, tf.sigmoid(ys_pre_a)], feed_dict={u1s_a: dtrain_a[0:800000, 0], u2s_a: dtrain_a[0:800000, 1], ys_a: np.float32(dtrain_a[0:800000, 2])})
        t_loss_a_v, t_target_loss_a_v, t_ys_a_v, t_ys_pre_a_v = sess.run([loss_a, target_loss_a, ys_a, tf.sigmoid(ys_pre_a)], feed_dict={u1s_a: dtest_a[0:800000, 0], u2s_a: dtest_a[0:800000, 1], ys_a: np.float32(dtest_a[0:800000, 2])})

        data_size_b = dtrain_b.shape[0]
        np.random.shuffle(dtrain_b)
        loss_b_v, target_loss_b_v, ys_b_v, ys_pre_b_v = sess.run([loss_b, target_loss_b, ys_b, tf.sigmoid(ys_pre_b)], feed_dict={u1s_b: dtrain_b[0:800000, 0], u2s_b: dtrain_b[0:800000, 1], ys_b: np.float32(dtrain_b[0:800000, 2])})

        print '----round%d: loss1: %f, target_loss1: %f, auc1: %f, t_loss1: %f, t_target_loss1: %f, t_auc1: %f ----loss2: %f, target_loss2: %f, auc2: %f' % (0, loss_a_v, target_loss_a_v, get_auc(ys_a_v, ys_pre_a_v), t_loss_a_v, t_target_loss_a_v, get_auc(t_ys_a_v, t_ys_pre_a_v), loss_b_v, target_loss_b_v, get_auc(ys_b_v, ys_pre_b_v))

        steps_of_round_a = data_size_a / params['batch_size']
        steps_of_round_b = data_size_b / params['batch_size']
        for i in range(params['round']):
            j = 0
            while (j < steps_of_round_b):
                start = params['batch_size'] * j
                end = params['batch_size'] * (j + 1)
                feed_dict = {u1s_b: dtrain_b[start:end, 0], u2s_b: dtrain_b[start:end, 1], ys_b: np.float32(dtrain_b[start:end, 2])}
                sess.run(train_step_b, feed_dict)
                j += 1
            loss_b_v, target_loss_b_v, ys_b_v, ys_pre_b_v = sess.run([loss_b, target_loss_b, ys_b, tf.sigmoid(ys_pre_b)], feed_dict={u1s_b: dtrain_b[0:800000, 0], u2s_b: dtrain_b[0:800000, 1], ys_b: np.float32(dtrain_b[0:800000, 2])})

            j=0
            while(j<steps_of_round_a):
                start = params['batch_size'] * j
                end = params['batch_size'] * (j + 1)
                feed_dict = {u1s_a: dtrain_a[start:end, 0], u2s_a: dtrain_a[start:end, 1], ys_a: np.float32(dtrain_a[start:end, 2])}
                sess.run(train_step_a, feed_dict)
                j+=1
            loss_a_v, target_loss_a_v, ys_a_v, ys_pre_a_v = sess.run([loss_a, target_loss_a, ys_a, tf.sigmoid(ys_pre_a)], feed_dict={u1s_a: dtrain_a[0:800000, 0], u2s_a: dtrain_a[0:800000, 1], ys_a: np.float32(dtrain_a[0:800000, 2])})
            t_loss_a_v, t_target_loss_a_v, t_ys_a_v, t_ys_pre_a_v = sess.run([loss_a, target_loss_a, ys_a, tf.sigmoid(ys_pre_a)], feed_dict={u1s_a: dtest_a[0:800000, 0], u2s_a: dtest_a[0:800000, 1], ys_a: np.float32(dtest_a[0:800000, 2])})

            print '----round%d: loss1: %f, target_loss1: %f, auc1: %f, t_loss1: %f, t_target_loss1: %f, t_auc1: %f ----loss2: %f, target_loss2: %f, auc2: %f' % (
                i+1, loss_a_v, target_loss_a_v, get_auc(ys_a_v, ys_pre_a_v), t_loss_a_v, t_target_loss_a_v, get_auc(t_ys_a_v, t_ys_pre_a_v), loss_b_v, target_loss_b_v, get_auc(ys_b_v, ys_pre_b_v))
            np.random.shuffle(dtrain_a)
            np.random.shuffle(dtest_a)
            np.random.shuffle(dtrain_b)

def get_auc(y_trues, y_pres):
    return roc_auc_score(y_trues, y_pres)
    # fpr, tpr, thresholds = metrics.roc_curve(np.int32(y_trues), y_pres, pos_label=1)
    # return metrics.auc(fpr, tpr)

def store_dict(dict,file_path):
    with open(file_path,'wb') as f:
        pickle.dump(dict,f,pickle.HIGHEST_PROTOCOL)

def load_dict(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path,'rb') as f:
        return pickle.load(f)

def main():
    # exp1({'dtrain_file_path': '../../data/test/small_train_data_v5_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v5_csv',
    #       'embedding_size': 20, 'node_num': 10000,
    #       'batch_size': 4800, 'step': 40 * 100, 'learning_rate': 5e-3, 'beta': 5e-1})#0.941
    # exp2({'dtrain_file_path': '../../data/test/small_train_data_v5_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v5_csv',
    #       'embedding_size': 20, 'node_num': 10000,
    #       'batch_size': 4800, 'step': 60 * 100, 'learning_rate': 5e-3, 'beta': 4e-1})#0.934
    # exp4({'dtrain_file_path': '../../data/test/small_train_data_v5_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v5_csv',
    #       'model_save_path': '../../data/test/small_data_pnn_model_params_saver_v2',
    #       'embedding_size': 20, 'node_num': 10000,
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 480, 'step': 30 * 1000, 'learning_rate': 4e-3, 'beta': 4e-4})#0.942
    # exp13({'dtrain_a_file_path': '../../data/test/small_train_data_v5_csv',
    #        'dtest_a_file_path': '../../data/test/small_test_data_v5_csv',
    #        'dtrain_b_file_path': '../../data/test/../../data/test/small_hop2_train_data_v2_csv',
    #        'pre_model_save_path': '../../data/test/small_data_pnn_model_params_saver_v2',
    #        'embedding_size': 20, 'node_num': 10000,
    #        'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size':20,
    #        'batch_size': 4800, 'round': 10, 'learning_rate': 4e-3, 'beta': 4e-4})#0.962

    # exp1({'dtrain_file_path': '../../data/test/hm_train_data_v5_csv',
    #       'dtest_file_path': '../../data/test/hm_test_data_v5_csv',
    #       'embedding_size': 20, 'node_num': 50000,
    #       'batch_size': 4000, 'step': 40 * 1000, 'learning_rate': 5e-3, 'beta': 5e-1})  # 0.947
    # exp2({'dtrain_file_path': '../../data/test/hm_train_data_v5_csv',
    #       'dtest_file_path': '../../data/test/hm_test_data_v5_csv',
    #       'embedding_size': 20, 'node_num': 50000,
    #       'batch_size': 4000, 'step': 60 * 1000, 'learning_rate': 5e-3, 'beta': 4e-1})  # 0.950
    # exp4({'dtrain_file_path': '../../data/test/hm_train_data_v5_csv',
    #       'dtest_file_path': '../../data/test/hm_test_data_v5_csv',
    #       'model_save_path': '../../data/test/hm_data_pnn_model_params_saver_v2',
    #       'embedding_size': 20, 'node_num': 50000,
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 4000, 'step': 30 * 1000, 'learning_rate': 4e-3, 'beta': 4e-4})  # 0.963
    # exp13({'dtrain_a_file_path': '../../data/test/hm_train_data_v5_csv',
    #        'dtest_a_file_path': '../../data/test/hm_test_data_v5_csv',
    #        'dtrain_b_file_path': '../../data/test/../../data/test/hm_hop2_train_data_v2_csv',
    #        'pre_model_save_path': '../../data/test/hm_data_pnn_model_params_saver_v2',
    #        'embedding_size': 20, 'node_num': 50000,
    #        'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #        'batch_size': 4800, 'round': 10, 'learning_rate': 4e-3, 'beta': 4e-4})  # 0.971

    # exp1({'dtrain_file_path': '../../data/test/eu_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/eu_test_data_v1_csv',
    #       'embedding_size': 20, 'node_num': 265214,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 5e-3, 'beta': 5e-1})#
    # exp2({'dtrain_file_path': '../../data/test/eu_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/eu_test_data_v1_csv',
    #       'embedding_size': 20, 'node_num': 265214,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 5e-3, 'beta': 4e-1})# 0.979
    # exp4({'dtrain_file_path': '../../data/test/eu_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/eu_test_data_v1_csv',
    #       'model_save_path': '../../data/test/eu_data_pnn_model_params_saver_v1',
    #       'embedding_size': 20, 'node_num': 265214,
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 4e-3, 'beta': 4e-4})#0.974
    # exp13({'dtrain_a_file_path': '../../data/test/eu_train_data_v1_csv',
    #        'dtest_a_file_path': '../../data/test/eu_test_data_v1_csv',
    #        'dtrain_b_file_path': '../../data/test/../../data/test/eu_hop2_train_data_v1_csv',
    #        'pre_model_save_path': '../../data/test/eu_data_pnn_model_params_saver_v1',
    #        'embedding_size': 20, 'node_num': 265214,
    #        'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #        'batch_size': 5000, 'round': 10, 'learning_rate': 4e-3, 'beta': 4e-4})

    # exp1({'dtrain_file_path': '../../data/test/enron_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/enron_test_data_v1_csv',
    #       'embedding_size': 20, 'node_num': 265214,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 5e-3, 'beta': 5e-1})# 0.941
    # exp2({'dtrain_file_path': '../../data/test/enron_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/enron_test_data_v1_csv',
    #       'embedding_size': 20, 'node_num': 265214,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 5e-3, 'beta': 4e-1})# 0.967
    # exp4({'dtrain_file_path': '../../data/test/enron_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/enron_test_data_v1_csv',
    #       'model_save_path': '../../data/test/enron_data_pnn_model_params_saver_v1',
    #       'embedding_size': 20, 'node_num': 265214,
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 4e-3, 'beta': 4e-4})
    # exp13({'dtrain_a_file_path': '../../data/test/enron_train_data_v1_csv',
    #        'dtest_a_file_path': '../../data/test/enron_test_data_v1_csv',
    #        'dtrain_b_file_path': '../../data/test/../../data/test/enron_hop2_train_data_v1_csv',
    #        'pre_model_save_path': '../../data/test/enron_data_pnn_model_params_saver_v1',
    #        'embedding_size': 20, 'node_num': 265214,
    #        'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #        'batch_size': 5000, 'round': 10, 'learning_rate': 4e-3, 'beta': 4e-4})

    # exp1({'dtrain_file_path': '../../data/test/cora_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/cora_test_data_v1_csv',
    #       'embedding_size': 20, 'node_num': 23166,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 5e-3, 'beta': 5e-1})# 0.917
    # exp2({'dtrain_file_path': '../../data/test/cora_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/cora_test_data_v1_csv',
    #       'embedding_size': 20, 'node_num': 23166,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 5e-3, 'beta': 4e-1})# 0.905
    # exp4({'dtrain_file_path': '../../data/test/cora_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/cora_test_data_v1_csv',
    #       'model_save_path': '../../data/test/cora_data_pnn_model_params_saver_v1',
    #       'embedding_size': 20, 'node_num': 23166,
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 4e-3, 'beta': 4e-3})#0.858
    # exp13({'dtrain_a_file_path': '../../data/test/cora_train_data_v1_csv',
    #        'dtest_a_file_path': '../../data/test/cora_test_data_v1_csv',
    #        'dtrain_b_file_path': '../../data/test/../../data/test/cora_hop2_train_data_v1_csv',
    #        'pre_model_save_path': '../../data/test/cora_data_pnn_model_params_saver_v1',
    #        'embedding_size': 20, 'node_num': 23166,
    #        'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #        'batch_size': 5000, 'round': 10, 'learning_rate': 4e-3, 'beta': 4e-3})#0.90

    # exp1({'dtrain_file_path': '../../data/test/epinions_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/epinions_test_data_v1_csv',
    #       'embedding_size': 20, 'node_num': 265214,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 5e-3, 'beta': 5e-1})  #0.91
    # exp2({'dtrain_file_path': '../../data/test/epinions_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/epinions_test_data_v1_csv',
    #       'embedding_size': 20, 'node_num': 265214,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 5e-3, 'beta': 4e-1})  #0.953
    # exp4({'dtrain_file_path': '../../data/test/epinions_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/epinions_test_data_v1_csv',
    #       'model_save_path': '../../data/test/epinions_data_pnn_model_params_saver_v1',
    #       'embedding_size': 20, 'node_num': 265214,
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 4e-3, 'beta': 4e-4})
    # exp13({'dtrain_a_file_path': '../../data/test/epinions_train_data_v1_csv',
    #        'dtest_a_file_path': '../../data/test/epinions_test_data_v1_csv',
    #        'dtrain_b_file_path': '../../data/test/../../data/test/epinions_hop2_train_data_v1_csv',
    #        'pre_model_save_path': '../../data/test/epinions_data_pnn_model_params_saver_v1',
    #        'embedding_size': 20, 'node_num': 265214,
    #        'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #        'batch_size': 5000, 'round': 10, 'learning_rate': 4e-3, 'beta': 4e-4})

    # exp1({'dtrain_file_path': '../../data/tmp/tmp_train_data_v1_csv',
    #       'dtest_file_path': '../../data/tmp/tmp_test_data_v1_csv',
    #       'embedding_size': 20, 'node_num': 2000,
    #       'batch_size': 6000, 'step': 30 * 100, 'learning_rate': 5e-3, 'beta': 5e-1})
    # exp2({'dtrain_file_path': '../../data/tmp/tmp_train_data_v1_csv',
    #       'dtest_file_path': '../../data/tmp/tmp_test_data_v1_csv',
    #       'embedding_size': 20, 'node_num': 2000,
    #       'batch_size': 6000, 'step': 30 * 100, 'learning_rate': 5e-3, 'beta': 4e-1})
    # exp4({'dtrain_file_path': '../../data/tmp/tmp_train_data_v1_csv',
    #       'dtest_file_path': '../../data/tmp/tmp_test_data_v1_csv',
    #       'model_save_path': '../../data/tmp/tmp_data_pnn_model_params_saver_v1',
    #       'embedding_size': 20, 'node_num': 2000,
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 6000, 'step': 30 * 100, 'learning_rate': 4e-3, 'beta': 4e-3})
    # exp13({'dtrain_a_file_path': '../../data/test/cora_train_data_v1_csv',
    #        'dtest_a_file_path': '../../data/test/cora_test_data_v1_csv',
    #        'dtrain_b_file_path': '../../data/test/../../data/test/cora_hop2_train_data_v1_csv',
    #        'pre_model_save_path': '../../data/test/cora_data_pnn_model_params_saver_v1',
    #        'embedding_size': 20, 'node_num': 265214,
    #        'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #        'batch_size': 5000, 'round': 10, 'learning_rate': 4e-3, 'beta': 4e-3})

    # exp1({'dtrain_file_path': '../../data/test/eu_train_data_v2_csv',
    #       'dtest_file_path': '../../data/test/eu_test_data_v2_csv',
    #       'embedding_size': 20, 'node_num': 265214,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 5e-3, 'beta': 5e-1})# 0.694
    # exp2({'dtrain_file_path': '../../data/test/eu_train_data_v2_csv',
    #       'dtest_file_path': '../../data/test/eu_test_data_v2_csv',
    #       'embedding_size': 20, 'node_num': 265214,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 5e-3, 'beta': 4e-1})# 0.979
    # exp4({'dtrain_file_path': '../../data/test/eu_train_data_v2_csv',
    #       'dtest_file_path': '../../data/test/eu_test_data_v2_csv',
    #       'model_save_path': '../../data/test/eu_data_pnn_model_params_saver_v2',
    #       'embedding_size': 20, 'node_num': 265214,
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 4e-3, 'beta': 4e-4})#0.974
    # exp13({'dtrain_a_file_path': '../../data/test/eu_train_data_v2_csv',
    #        'dtest_a_file_path': '../../data/test/eu_test_data_v2_csv',
    #        'dtrain_b_file_path': '../../data/test/../../data/test/eu_hop2_train_data_v2_csv',
    #        'pre_model_save_path': '../../data/test/eu_data_pnn_model_params_saver_v2',
    #        'embedding_size': 20, 'node_num': 265214,
    #        'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #        'batch_size': 5000, 'round': 10, 'learning_rate': 4e-3, 'beta': 4e-4})

    # exp1({'dtrain_file_path': '../../data/test/cora_train_data_v2_csv',
    #       'dtest_file_path': '../../data/test/cora_test_data_v2_csv',
    #       'embedding_size': 20, 'node_num': 23166,
    #       'batch_size': 3000, 'step': 30 * 1000, 'learning_rate': 5e-3, 'beta': 5e-1})  # 0.863
    # exp2({'dtrain_file_path': '../../data/test/cora_train_data_v2_csv',
    #       'dtest_file_path': '../../data/test/cora_test_data_v2_csv',
    #       'embedding_size': 20, 'node_num': 23166,
    #       'batch_size': 3000, 'step': 30 * 1000, 'learning_rate': 5e-3, 'beta': 4e-1})  # 0.875
    # exp4({'dtrain_file_path': '../../data/test/cora_train_data_v2_csv',
    #       'dtest_file_path': '../../data/test/cora_test_data_v2_csv',
    #       'model_save_path': '../../data/test/cora_data_pnn_model_params_saver_v2',
    #       'embedding_size': 20, 'node_num': 23166,
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 1600, 'step': 30 * 1000, 'learning_rate': 4e-3, 'beta': 4e-4})  # 0.917
    # exp13({'dtrain_a_file_path': '../../data/test/cora_train_data_v2_csv',
    #        'dtest_a_file_path': '../../data/test/cora_test_data_v2_csv',
    #        'dtrain_b_file_path': '../../data/test/../../data/test/cora_hop2_train_data_v2_csv',
    #        'pre_model_save_path': '../../data/test/cora_data_pnn_model_params_saver_v2',
    #        'embedding_size': 20, 'node_num': 23166,
    #        'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #        'batch_size': 5000, 'round': 10, 'learning_rate': 4e-3, 'beta': 4e-4}) # 0.934

    # exp1({'dtrain_file_path': '../../data/test/enron_train_data_v2_csv',
    #       'dtest_file_path': '../../data/test/enron_test_data_v2_csv',
    #       'embedding_size': 20, 'node_num': 36692,
    #       'batch_size': 4800, 'step': 30 * 2500, 'learning_rate': 5e-3, 'beta': 5e-1})  # 0.857
    # exp2({'dtrain_file_path': '../../data/test/enron_train_data_v2_csv',
    #       'dtest_file_path': '../../data/test/enron_test_data_v2_csv',
    #       'embedding_size': 20, 'node_num': 36692,
    #       'batch_size': 4800, 'step': 30 * 2500, 'learning_rate': 5e-3, 'beta': 4e-1})  # 0.918
    # exp4({'dtrain_file_path': '../../data/test/enron_train_data_v2_csv',
    #       'dtest_file_path': '../../data/test/enron_test_data_v2_csv',
    #       'model_save_path': '../../data/test/enron_data_pnn_model_params_saver_v2',
    #       'embedding_size': 20, 'node_num': 36692,
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 4800, 'step': 30 * 2500, 'learning_rate': 4e-3, 'beta': 4e-4})  # 0.990
    # exp13({'dtrain_a_file_path': '../../data/test/enron_train_data_v2_csv',
    #        'dtest_a_file_path': '../../data/test/enron_test_data_v2_csv',
    #        'dtrain_b_file_path': '../../data/test/../../data/test/enron_hop2_train_data_v2_csv',
    #        'pre_model_save_path': '../../data/test/enron_data_pnn_model_params_saver_v2',
    #        'embedding_size': 20, 'node_num': 36692,
    #        'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #        'batch_size': 4800, 'round': 10, 'learning_rate': 4e-3, 'beta': 4e-4}) #0.992

    # exp0({'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #       'node_num': 10000, 'embedding_size':20,
    #       'h1_size': 20, 'h2_size': 20,
    #       'batch_size': 4800, 'step': 30*100,
    #       'learning_rate': 3e-3, 'beta': 3e-3})#can not train it well
    # exp1({'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #       'embedding_size': 20, 'node_num': 10000,
    #       'batch_size': 4800, 'step': 30 * 100, 'learning_rate': 5e-3, 'beta': 5e-1})#auc: 0.920
    # exp1({'dtrain_file_path': '../../data/test/m_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/m_test_data_v1_csv',
    #       'embedding_size': 20, 'node_num': 100000,
    #       'batch_size': 8000, 'step': 30 * 1000, 'learning_rate': 5e-3, 'beta': 5e-1})#auc: 0.952
    # exp2({'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #       'embedding_size': 20, 'node_num': 10000,
    #       'batch_size': 4800, 'step': 60 * 100, 'learning_rate': 5e-3, 'beta': 6e-1})#auc: 0.905
    # exp2({'dtrain_file_path': '../../data/test/m_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/m_test_data_v1_csv',
    #       'embedding_size': 20, 'node_num': 100000,
    #       'batch_size': 8000, 'step': 30 * 1000, 'learning_rate': 5e-3, 'beta': 6e-1})#auc: 0.938
    # exp3({'dtrain_file_path': '../../data/test/m_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/m_test_data_v1_csv',
    #       'node_num': 100000, 'embedding_size': 20, 'h1_size': 8, 'h2_size': 0, 'batch_size': 8000, 'step': 30*1000,
    #       'learning_rate': 1e-2, 'learning_rate1': 1e-2, 'learning_rate2': 3e-7, 'beta': 1e-4})#auc: 0.949
    # exp4({'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #        'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #        'model_save_path': '../../data/test/small_data_pnn_model_params_saver',
    #        'embedding_size': 20, 'node_num': 10000,
    #        'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size':20,
    #        'batch_size': 480, 'step': 30*1000, 'learning_rate': 5e-3, 'beta': 2e-4})#auc: 0.926
    # exp4({'dtrain_file_path': '../../data/test/m_train_data_v1_csv',
    #       'dtest_file_path': '../../data/test/m_test_data_v1_csv',
    #       'model_save_path': '../../data/test/m_data_pnn_model_params_saver',
    #       'embedding_size': 20, 'node_num': 100000,
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 8000, 'step': 20 * 1000, 'learning_rate': 5e-3, 'beta': 2e-4})#auc: 0.968
    # exp5({'dtrain_file_path': '../../data/test/m_train_data_v2_csv',
    #       'dtest_file_path': '../../data/test/m_test_data_v2_csv',
    #       'embedding_size': 20, 'node_num': 100000, 'feature_depth': [100000, 2, 50, 2, 30, 100, 100],
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size':20,
    #       'batch_size': 8000, 'step': 10 * 1000, 'learning_rate': 3e-3, 'beta': 0.2})
    # exp5({'dtrain_file_path': '../../data/test/m_train_data_v2_csv',
    #       'dtest_file_path': '../../data/test/m_test_data_v2_csv',
    #       'embedding_size': 10, 'node_num': 100000, 'feature_depth': [100000, 2, 50, 2, 30, 100, 100],
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 8000, 'step': 10 * 1000, 'learning_rate': 3e-3, 'beta': 0.1})
    # exp6({'dtrain_file_path': '../../data/test/small_train_data_v4_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v4_csv',
    #       'pre_model_save_path': '../../data/test/small_data_pnn_model_params_saver',
    #       'embedding_size': 20, 'node_num': 10000, 'feature_depth': [10000, 2, 50, 2, 30, 100, 100],
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 4800, 'step': 30 * 100, 'learning_rate': 5e-3, 'beta': 5e-2})#auc: 0.935
    # exp6({'dtrain_file_path': '../../data/test/m_train_data_v2_csv',
    #       'dtest_file_path': '../../data/test/m_test_data_v2_csv',
    #       'pre_model_save_path': '../../data/test/m_data_pnn_model_params_saver',
    #       'embedding_size': 20, 'node_num': 100000, 'feature_depth': [100000, 2, 50, 2, 30, 100, 100],
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 8000, 'step': 10 * 1000, 'learning_rate': 3e-3, 'beta': 3e-2})#auc: 0.971
    # exp7({'dtrain_file_path': '../../data/test/small_train_data_v4_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v4_csv',
    #       'pre_model_save_path': '../../data/test/small_data_pnn_model_params_saver',
    #       'embedding_size': 20, 'node_num': 10000, 'feature_depth': [10000, 2, 50, 2, 30, 100, 100],
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 4800, 'step': 20 * 100, 'learning_rate': 3e-3, 'beta': 3e-2})#auc:0.934, 'learning_rate': 5e-3, 'beta': 5e-2
    # exp7({'dtrain_file_path': '../../data/test/m_train_data_v2_csv',
    #       'dtest_file_path': '../../data/test/m_test_data_v2_csv',
    #       'pre_model_save_path': '../../data/test/m_data_pnn_model_params_saver',
    #       'embedding_size': 20, 'node_num': 100000, 'feature_depth': [100000, 2, 50, 2, 30, 100, 100],
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 8000, 'step': 20 * 1000, 'learning_rate': 3e-3, 'beta': 3e-2})#auc: 0.971

    # exp8({'dtrain_file_path': '../../data/test/small_train_data_v4_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v4_csv',
    #       'pre_model_save_path': '../../data/test/small_data_pnn_model_params_saver',
    #       'np_rate': 3,
    #       'embedding_size': 20, 'node_num': 10000, 'feature_depth': [10000, 2, 50, 2, 30, 100, 100],
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'ph1_size': 20, 'ph2_size': 20, 'ph3_size': 20, 'ph4_size': 20,
    #       'batch_size': 8000, 'step': 20 * 1000, 'learning_rate': 1e-2, 'beta': 4e-2, 'pbeta': 4e-2, 'lamda': 1e-1})

    # exp13({'dtrain_a_file_path': '../../data/test/m_train_data_v1_csv',
    #       'dtest_a_file_path': '../../data/test/m_test_data_v1_csv',
    #       'pre_model_save_path': '../../data/test/m_data_pnn_model_params_saver',
    #       'embedding_size': 20, 'node_num': 100000,
    #       'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
    #       'batch_size': 8000, 'round': 4, 'learning_rate': 5e-3, 'beta': 2e-4})

    exp1({'dtrain_file_path': '../../data/test1/cora_train_data_v1.csv',
          'dtest_file_path': '../../data/test1/cora_test_data_v1.csv',
          'embedding_size': 20, 'node_num': 10000,
          'batch_size': 4800, 'step': 40 * 100, 'learning_rate': 5e-3, 'beta': 5e-1})#0.941
    exp2({'dtrain_file_path': '../../data/test1/cora_train_data_v1.csv',
          'dtest_file_path': '../../data/test1/cora_test_data_v1.csv',
          'embedding_size': 20, 'node_num': 10000,
          'batch_size': 4800, 'step': 60 * 100, 'learning_rate': 5e-3, 'beta': 4e-1})#0.934
    exp4({'dtrain_file_path': '../../data/test1/cora_train_data_v1.csv',
          'dtest_file_path': '../../data/test1/cora_test_data_v1.csv',
          'model_save_path': '../../data/test1/cora_data_pnn_model_params_saver_v1',
          'embedding_size': 20, 'node_num': 10000,
          'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size': 20,
          'batch_size': 480, 'step': 30 * 1000, 'learning_rate': 4e-3, 'beta': 4e-4})#0.942
    exp13({'dtrain_a_file_path': '../../data/test1/cora_train_data_v1.csv',
           'dtest_a_file_path': '../../data/test1/cora_test_data_v1.csv',
           'dtrain_b_file_path': '../../data/test1/cora_hop2_train_data_v1.csv',
           'pre_model_save_path': '../../data/test1/cora_data_pnn_model_params_saver_v1',
           'embedding_size': 20, 'node_num': 10000,
           'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'h4_size':20,
           'batch_size': 4800, 'round': 10, 'learning_rate': 4e-3, 'beta': 4e-4})#0.962

if __name__ == '__main__':
    main()