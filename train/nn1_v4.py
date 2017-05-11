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

def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

def gen_predict_result(ypred,test_data_file_path,target_file_path,log_file_path):
    #result format:tag\tpred\tcall_count\n
    with open(test_data_file_path,'r') as read_file,open(target_file_path,'w') as write_file,open(log_file_path,'a') as log_file:
        readlines=read_file.readlines()
        result_list=[]
        for i in range(0,len(readlines)):
            items=readlines[i][0:len(readlines[i])-1].split(' ')
            newline='%s\t%.3f\t%s\n'%(items[0],ypred[i],items[5].split(':')[1])
            result_list.append(newline)
        write_file.writelines(result_list)
        localtime=time.asctime(time.localtime(time.time()))
        log='result_file: %s, result_lines: %d, time: %s\n'%(target_file_path,len(readlines),localtime)
        print log
        log_file.write(log)

def xgboost(train_data_file_path, test_data_file_path, param, num_round):
    dtrain = xgb.DMatrix(train_data_file_path)
    dtest = xgb.DMatrix(test_data_file_path)
    plst = param.items()
    bst = xgb.train(plst, dtrain, num_round)
    ypred = bst.predict(dtest)
    ytrue = np.int32(np.loadtxt(test_data_file_path, dtype='str', delimiter = ' ')[:,0])
    print get_auc(ytrue, ypred)

def mf_only_with_biases(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter = ',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter = ',')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], 0.0, 0.2))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], 0.0, 0.2))

    u1s = tf.placeholder(tf.int32, shape = [None])
    u2s = tf.placeholder(tf.int32, shape = [None])
    ys = tf.placeholder(tf.float32, shape = [None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    ys_pre = tf.sigmoid(tf.reduce_sum(e1+e2, 1))#tf.sigmoid(tf.reduce_sum(dot_e, 1))

    loss = tf.reduce_mean(0.5 * tf.square(ys - ys_pre) + params['beta'] * (tf.reduce_sum(tf.square(e1) + tf.square(e2), 1)))

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss) # tell me why GradientDescentOptimizer did not work.........

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - ys_pre)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.random.shuffle(dtrain)
        data_size = dtrain.shape[0]
        a_rmse = 0
        a_loss = 0
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i+1) % data_size
            if end <= start:
                continue
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            _, rmse_v, loss_v = sess.run([train_step, rmse, loss], feed_dict)
            a_rmse += rmse_v
            a_loss += loss_v
            if i != 0 and i%100 == 0:
                a_rmse /= 100
                a_loss /= 100
                print 'step%d: %f, %f' % (i, a_rmse, a_loss)
                a_rmse = 0
                a_loss = 0
                ys_pre_v, ys_true = sess.run([ys_pre, ys], {u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print 'test auc: %f' % get_auc(ys_true, ys_pre_v)
        ys_pre_v, ys_true = sess.run([ys_pre, ys], {u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test auc: %f'%get_auc(ys_true, ys_pre_v)

def mf_with_nn(params):
    dtrain = np.loadtxt(params['source_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], 0.0, 0.2))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], 0.0, 0.2))

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    e_cc = tf.concat(1, [e1, e2, e1*e2])

    weights1 = tf.Variable(tf.ones([params['embedding_size'] * 3, params['h1_size']]))
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(e_cc, weights1) + biases1

    weights2 = tf.Variable(tf.ones([params['h1_size'], 1]))
    biases2 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.sigmoid(tf.matmul(h1, weights2) + biases2))
    ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)

    # loss = tf.reduce_mean(0.5 * tf.square(ys - ys_pre) + params['beta'] * (tf.reduce_sum(tf.square(e1) + tf.square(e2), 1)))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(
        loss)  # tell me why GradientDescentOptimizer did not work.........

    # rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - ys_pre)))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(ys_, tf.int32)), tf.float32))
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        np.random.shuffle(dtrain)
        data_size = dtrain.shape[0]
        a_rmse = 0
        a_loss = 0
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i+1) % data_size
            if end <= start:
                continue
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            _, rmse_v, loss_v, weights1_v, biases1_v, weights2_v, biases2_v = sess.run([train_step, rmse, loss, weights1, biases1, weights2, biases2], feed_dict)
            a_rmse += rmse_v
            a_loss += loss_v
            if i != 0 and i%100 == 0:
                a_rmse /= 100
                a_loss /= 100
                print '----step%d: %f, %f' % (i, a_rmse, a_loss)
                # print weights1_v, biases1_v, weights2_v, biases2_v
                a_rmse = 0
                a_loss = 0
        y_trues, y_pres, accuracy_v = sess.run([ys, ys_pre, accuracy], feed_dict = {u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print roc_auc_score(y_trues, y_pres), accuracy_v
        # for i in range(1000):
        #     # print y_trues[i], y_pres[i]
        #     print y_trues[i], sigmoid(y_pres[i])

def mf_with_weight(params):
    #train rmse : 0.04
    dtrain = np.loadtxt(params['source_file_path'], delimiter = ',')
    dtrain -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], 0.0, 0.2))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], 0.0, 0.2))

    u1s = tf.placeholder(tf.int32, shape = [None])
    u2s = tf.placeholder(tf.int32, shape = [None])
    ys = tf.placeholder(tf.float32, shape = [None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    weights1 = tf.Variable(tf.ones([params['embedding_size'],params['h1_size']]))
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(dot_e, weights1) + biases1

    weights2 = tf.Variable(tf.ones([params['h1_size'], 1]))
    biases2 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.sigmoid(tf.matmul(h1, weights2) + biases2))
    ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)

    # loss = tf.reduce_mean(0.5 * tf.square(ys - ys_pre) + params['beta'] * (tf.reduce_sum(tf.square(e1) + tf.square(e2), 1)))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss) # tell me why GradientDescentOptimizer did not work.........

    # rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - ys_pre)))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(ys_, tf.int32)), tf.float32))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys,tf.int32),tf.cast(tf.sigmoid(ys_pre),tf.int32)),tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        np.random.shuffle(dtrain)
        data_size = dtrain.shape[0]
        a_rmse = 0
        a_loss = 0
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i+1) % data_size
            if end <= start:
                continue
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            _, rmse_v, loss_v, weights1_v, biases1_v, weights2_v, biases2_v = sess.run([train_step, rmse, loss, weights1, biases1, weights2, biases2], feed_dict)
            a_rmse += rmse_v
            a_loss += loss_v
            if i != 0 and i%100 == 0:
                a_rmse /= 100
                a_loss /= 100
                print '----step%d: %f, %f' % (i, a_rmse, a_loss)
                # print weights1_v, biases1_v, weights2_v, biases2_v
                a_rmse = 0
                a_loss = 0
        y_trues, y_pres, accuracy_v = sess.run([ys, ys_pre, accuracy], feed_dict = {u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print roc_auc_score(y_trues, y_pres), accuracy_v
        # for i in range(1000):
        #     # print y_trues[i], y_pres[i]
        #     print y_trues[i], sigmoid(y_pres[i])

def mf(params):
    #train rmse : 0.177 test auc :
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter = ',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter = ',')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], 0.0, 0.2))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], 0.0, 0.2))

    u1s = tf.placeholder(tf.int32, shape = [None])
    u2s = tf.placeholder(tf.int32, shape = [None])
    ys = tf.placeholder(tf.float32, shape = [None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    ys_pre = tf.sigmoid(tf.reduce_sum(dot_e, 1))

    loss = tf.reduce_mean(0.5 * tf.square(ys - ys_pre) + params['beta'] * (tf.reduce_sum(tf.square(e1) + tf.square(e2), 1)))

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss) # tell me why GradientDescentOptimizer did not work.........

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - ys_pre)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.random.shuffle(dtrain)
        data_size = dtrain.shape[0]
        a_rmse = 0
        a_loss = 0
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i+1) % data_size
            if end <= start:
                continue
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            _, rmse_v, loss_v = sess.run([train_step, rmse, loss], feed_dict)
            a_rmse += rmse_v
            a_loss += loss_v
            if i != 0 and i%100 == 0:
                a_rmse /= 100
                a_loss /= 100
                print 'step%d: %f, %f' % (i, a_rmse, a_loss)
                a_rmse = 0
                a_loss = 0
                ys_pre_v, ys_true = sess.run([ys_pre, ys], {u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print 'test auc: %f' % get_auc(ys_true, ys_pre_v)
        ys_pre_v, ys_true = sess.run([ys_pre, ys], {u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test auc: %f'%get_auc(ys_true, ys_pre_v)

def gen_svd_embeddings(source_file_path, target_file_path):
    #train rmse : 0.235
    with open(source_file_path,'r') as read_file:
        embedding_size=4
        step=100
        alpha=0.1
        belta=0.0002
        node_num=10000
        p=np.random.uniform(0.0, 0.2, size=(node_num, embedding_size))
        q=np.random.uniform(0.0, 0.2, size=(node_num, embedding_size))
        read_lines=read_file.readlines()

        for s in range(0,step):
            count=0
            f_loss=0
            for line in read_lines:
                count+=1
                if count%10000==0:
                    print('step:%d,count:%dw'%(s,count/10000))
                items=line[0:len(line)-1].split('\t')
                i=int(items[0])-1
                j=int(items[1])-1

                r_ij = 0
                for l in range(0, embedding_size):
                    r_ij += p[i][l] * q[j][l]
                r_ij = sigmoid(r_ij)
                f_loss += math.pow(1.0 - r_ij, 2)
                for h in range(0,embedding_size):
                    p[i][h]=(1-alpha*belta)*p[i][h]+alpha*q[j][h]*(1-r_ij)*(1-r_ij)*r_ij
                    q[j][h]=(1-alpha*belta)*q[j][h]+alpha*p[i][h]*(1-r_ij)*(1-r_ij)*r_ij
                for o in range(0,4):
                    i=randint(0,node_num-1)
                    j=randint(0,node_num-1)

                    r_ij = 0
                    for l in range(0, embedding_size):
                        r_ij += p[i][l] * q[j][l]
                    r_ij = sigmoid(r_ij)
                    f_loss += math.pow(0.0 - r_ij, 2)

                    for h in range(0,embedding_size):
                        p[i][h]=(1-alpha*belta)*p[i][h]+alpha*q[j][h]*(0-r_ij)*(1-r_ij)*r_ij
                        q[j][h]=(1-alpha*belta)*q[j][h]+alpha*p[i][h]*(0-r_ij)*(1-r_ij)*r_ij
            print 'rmse:%f' % math.sqrt((f_loss/(5*len(read_lines))))
        save_dict = {'p': p, 'q': q}
        store_dict(save_dict, target_file_path)

        # V = np.zeros([node_num, node_num])
        # for line in read_lines:
        #     items = line[0:len(line) - 1].split('\t')
        #     i = int(items[0]) - 1
        #     j = int(items[1]) - 1
        #     V[i][j] = 1
        # print V
        # print [[sigmoid(xx) for xx in x] for x in np.matmul(p, q.T)]

def mf_with_weight_for_debug(params):
    dtrain = np.loadtxt(params['source_file_path'], delimiter = ',')
    dtrain -= [1, 1, 0]

    pq_dict = load_dict(params['pq_file_path'])
    p = pq_dict['p']
    q = pq_dict['q']

    u1s = tf.placeholder(tf.int32, shape = [None])
    u2s = tf.placeholder(tf.int32, shape = [None])
    ys = tf.placeholder(tf.float32, shape = [None])

    embeddings1 = tf.Variable(np.float32(p), name='embeddings1')
    embeddings2 = tf.Variable(np.float32(q), name='embeddings2')
    e_biases = tf.Variable(tf.zeros([params['embedding_size']]), name='e_biases')
    e1 = tf.nn.embedding_lookup(embeddings1, u1s)
    e2 = tf.nn.embedding_lookup(embeddings2, u2s)
    e3 = tf.mul(e1, e2)
    h0 = e3 + e_biases  # tf.concat(1, [e1, e2, e3]) + e_biases

    weights1 = tf.Variable(tf.ones([params['embedding_size'],params['h1_size']]))
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(h0, weights1) + biases1

    weights2 = tf.Variable(tf.ones([params['h1_size'], 1]))
    biases2 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.sigmoid(tf.matmul(h1, weights2) + biases2))
    ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)

    # loss = tf.reduce_mean(0.5 * tf.square(ys - ys_pre) + params['beta'] * (tf.reduce_sum(tf.square(e1) + tf.square(e2), 1)))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss) # tell me why GradientDescentOptimizer did not work.........

    # rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - ys_pre)))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(ys_, tf.int32)), tf.float32))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys,tf.int32),tf.cast(tf.sigmoid(ys_pre),tf.int32)),tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        np.random.shuffle(dtrain)
        data_size = dtrain.shape[0]
        a_rmse = 0
        a_loss = 0
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i+1) % data_size
            if end <= start:
                continue
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            _, rmse_v, loss_v, weights1_v, biases1_v, weights2_v, biases2_v = sess.run([train_step, rmse, loss, weights1, biases1, weights2, biases2], feed_dict)
            a_rmse += rmse_v
            a_loss += loss_v
            if i != 0 and i%100 == 0:
                a_rmse /= 100
                a_loss /= 100
                print '----step%d: %f, %f' % (i, a_rmse, a_loss)
                # print weights1_v, biases1_v, weights2_v, biases2_v
                a_rmse = 0
                a_loss = 0
        y_trues, y_pres, accuracy_v = sess.run([ys, ys_pre, accuracy], feed_dict = {u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print roc_auc_score(y_trues, y_pres), accuracy_v
        # for i in range(1000):
        #     # print y_trues[i], y_pres[i]
        #     print y_trues[i], sigmoid(y_pres[i])

def exp1(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], 0.0, 0.2))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], 0.0, 0.2))

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    weights1 = tf.Variable(tf.ones([params['embedding_size'], params['h1_size']]))
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(dot_e, weights1) + biases1

    weights2 = tf.Variable(tf.ones([params['h1_size'], 1]))
    biases2 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        np.random.shuffle(dtrain)
        data_size = dtrain.shape[0]
        a_rmse = 0
        a_loss = 0
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            if i % 100 == 0:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)

def exp2(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))

    features = tf.constant(feature_list)

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    f1 = tf.nn.embedding_lookup(features, u1s)
    f2 = tf.nn.embedding_lookup(features, u2s)

    ff = tf.concat(1, [f1, f2])

    weights1_1 = tf.Variable(tf.random_uniform([params['embedding_size'], params['h1_size']], -0.1, 0.1))#(tf.zeros([params['embedding_size'], params['h1_size']]))#
    weights1_2 = tf.Variable(tf.random_uniform([feature_list.shape[1] * 2, params['h1_size']], -0.1, 0.1))#(tf.zeros([feature_list.shape[1] * 2, params['h1_size']]))#(tf.random_uniform([feature_list.shape[1] * 2, params['h1_size']], 0, 0))
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(dot_e, weights1_1) + tf.matmul(ff, weights1_2) + biases1

    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.01, 0.01))
    # biases2 = tf.Variable(tf.zeros([params['h2_size']]))
    # h2 = tf.matmul(h1, weights2) + biases2
    #
    # weights3 = tf.Variable(tf.random_uniform([params['h2_size'], 1], -0.01, 0.01))
    # biases3 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h2, weights3) + biases3)

    weights2 = tf.Variable(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    biases2 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys))# + params['beta'] * (tf.reduce_sum(tf.square(weights1_1)) + tf.reduce_sum(tf.square(weights1_2)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))

    # train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    var_list1 = [p, q]
    var_list2 = [weights1_1, weights1_2, weights2, biases1, biases2]
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
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                tmp, rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([weights2, rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#input layer: dot_e
def exp3(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))

    features = tf.constant(feature_list)

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    f1 = tf.nn.embedding_lookup(features, u1s)
    f2 = tf.nn.embedding_lookup(features, u2s)

    ff = tf.concat(1, [f1, f2])

    weights1_1 = tf.Variable(tf.random_uniform([params['embedding_size'], params['h1_size']], -0.1, 0.1))#(tf.zeros([params['embedding_size'], params['h1_size']]))#
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(dot_e, weights1_1) + biases1

    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.01, 0.01))
    # biases2 = tf.Variable(tf.zeros([params['h2_size']]))
    # h2 = tf.matmul(h1, weights2) + biases2
    #
    # weights3 = tf.Variable(tf.random_uniform([params['h2_size'], 1], -0.01, 0.01))
    # biases3 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h2, weights3) + biases3)

    weights2 = tf.Variable(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    biases2 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_1)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))

    # train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    var_list1 = [p, q]
    var_list2 = [weights1_1, weights2, biases1, biases2]
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
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                tmp, rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([weights2, rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#input layer: concat(dot_e, ff)
def exp4(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))

    features = tf.constant(feature_list)

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    f1 = tf.nn.embedding_lookup(features, u1s)
    f2 = tf.nn.embedding_lookup(features, u2s)
    ff = tf.concat(1, [f1, f2])

    weights1_1 = tf.Variable(tf.random_uniform([params['embedding_size'], params['h1_size']], -0.1, 0.1))#(tf.zeros([params['embedding_size'], params['h1_size']]))#
    weights1_2 = tf.Variable(tf.random_uniform([feature_list.shape[1] * 2, params['h1_size']], -1, 1))#(tf.zeros([feature_list.shape[1] * 2, params['h1_size']]))#(tf.random_uniform([feature_list.shape[1] * 2, params['h1_size']], 0, 0))
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 =tf.matmul(dot_e, weights1_1) +  tf.nn.softmax(tf.matmul(ff, weights1_2)) + biases1
    # h1 = tf.matmul(dot_e, weights1_1) + biases1

    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.01, 0.01))
    # biases2 = tf.Variable(tf.zeros([params['h2_size']]))
    # h2 = tf.matmul(h1, weights2) + biases2
    #
    # weights3 = tf.Variable(tf.random_uniform([params['h2_size'], 1], -0.01, 0.01))
    # biases3 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h2, weights3) + biases3)

    weights2 = tf.Variable(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'] + 2 * feature_list.shape[1], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    biases2 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)
    # ys_pre = tf.squeeze(tf.matmul(tf.concat(1, [h1, ff]), weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_1)) + tf.reduce_sum(tf.square(weights1_2)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_1)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))

    # train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    # var_list1 = [p, q, weights1_2]
    var_list1 = [p, q]
    var_list2 = [weights1_1, weights2, biases1, biases2]
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
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#input layer: dot_e, concat(h1, ff)
def exp5(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))

    features = tf.constant(feature_list)

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    f1 = tf.nn.embedding_lookup(features, u1s)
    f2 = tf.nn.embedding_lookup(features, u2s)
    ff = tf.concat(1, [f1, f2])

    weights1 = tf.Variable(tf.random_uniform([params['embedding_size'], params['h1_size']], -0.1, 0.1))#(tf.zeros([params['embedding_size'], params['h1_size']]))#
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(dot_e, weights1) + biases1

    weights2_1 = tf.Variable(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    weights2_2 = tf.Variable(tf.random_uniform([2 * feature_list.shape[1], 1], -0.1, 0.1))
    biases2 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h1, weights2_1) + tf.matmul(ff, weights2_2) + biases2)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1)) + tf.reduce_sum(tf.square(weights2_1)) + tf.reduce_sum(tf.square(weights2_2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))

    # train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    # var_list1 = [p, q, weights1_2]
    var_list1 = [p, q]
    var_list2 = [weights1, weights2_1, weights2_2, biases1, biases2]
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
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#input layer: ff
def exp6(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))

    features = tf.constant(feature_list)

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    f1 = tf.nn.embedding_lookup(features, u1s)
    f2 = tf.nn.embedding_lookup(features, u2s)
    ff = tf.concat(1, [f1, f2])

    weights1_2 = tf.Variable(tf.random_uniform([feature_list.shape[1] * 2, params['h1_size']], -1, 1))#(tf.zeros([feature_list.shape[1] * 2, params['h1_size']]))#(tf.random_uniform([feature_list.shape[1] * 2, params['h1_size']], 0, 0))
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(ff, weights1_2) + biases1
    # h1 = tf.matmul(dot_e, weights1_1) + biases1

    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.01, 0.01))
    # biases2 = tf.Variable(tf.zeros([params['h2_size']]))
    # h2 = tf.matmul(h1, weights2) + biases2
    #
    # weights3 = tf.Variable(tf.random_uniform([params['h2_size'], 1], -0.01, 0.01))
    # biases3 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h2, weights3) + biases3)

    weights2 = tf.Variable(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'] + 2 * feature_list.shape[1], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    biases2 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)
    # ys_pre = tf.squeeze(tf.matmul(tf.concat(1, [h1, ff]), weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_2)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_1)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    # var_list1 = [p, q, weights1_2]
    # var_list1 = [p, q]
    # var_list2 = [weights1_1, weights2, biases1, biases2]
    # op1 = tf.train.AdamOptimizer(params['learning_rate1']).minimize(loss, var_list = var_list1)
    # op2 = tf.train.AdamOptimizer(params['learning_rate2']).minimize(loss, var_list = var_list2)
    # train_step = tf.group(op1, op2)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        np.random.shuffle(dtrain)
        data_size = dtrain.shape[0]
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#input layer: dot_e, 2 hidden layer
def exp7(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))

    features = tf.constant(feature_list)

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    weights1_1 = tf.Variable(tf.random_uniform([params['embedding_size'], params['h1_size']], -0.1, 0.1))#(tf.zeros([params['embedding_size'], params['h1_size']]))#
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(dot_e, weights1_1) + biases1

    weights2 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.1, 0.1))
    biases2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.matmul(h1, weights2) + biases2

    # weights3 = tf.Variable(tf.random_uniform([params['h2_size'], params['h3_size']], -0.1, 0.1))
    # biases3 = tf.Variable(tf.zeros([params['h3_size']]))
    # h3 = tf.matmul(h2, weights3) + biases3

    weights3 = tf.Variable(tf.random_uniform([params['h2_size'], 1], -0.1, 0.1))
    biases3 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h2, weights3) + biases3)

    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    # biases2 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_1)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))

    # train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    var_list1 = [p, q]
    var_list2 = [weights1_1, weights2, biases1, biases2]
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
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                tmp, rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([weights2, rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#input layer: bucket_ff, concat(buckets)
def exp8(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    # features = tf.constant(feature_list)
    t0 = tf.one_hot(feature_list[:, 0], params['bucket_dims'][0])
    t1 = tf.one_hot(feature_list[:, 1], params['bucket_dims'][1])
    t2 = tf.one_hot(feature_list[:, 2], params['bucket_dims'][2])
    t3 = tf.one_hot(feature_list[:, 3], params['bucket_dims'][3])
    t4 = tf.one_hot(feature_list[:, 4], params['bucket_dims'][4])
    t5 = tf.one_hot(feature_list[:, 5], params['bucket_dims'][5])
    features = tf.concat(1, [t0, t1, t2, t3, t4, t5])

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    f1 = tf.nn.embedding_lookup(features, u1s)
    f2 = tf.nn.embedding_lookup(features, u2s)
    ff = tf.concat(1, [f1, f2])

    weights1_2 = tf.Variable(tf.random_uniform([np.sum(params['bucket_dims']) * 2, params['h1_size']], -0.1, 0.1))#(tf.zeros([feature_list.shape[1] * 2, params['h1_size']]))#(tf.random_uniform([feature_list.shape[1] * 2, params['h1_size']], 0, 0))
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(ff, weights1_2) + biases1
    # h1 = tf.matmul(dot_e, weights1_1) + biases1

    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.01, 0.01))
    # biases2 = tf.Variable(tf.zeros([params['h2_size']]))
    # h2 = tf.matmul(h1, weights2) + biases2
    #
    # weights3 = tf.Variable(tf.random_uniform([params['h2_size'], 1], -0.01, 0.01))
    # biases3 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h2, weights3) + biases3)

    weights2 = tf.Variable(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'] + 2 * feature_list.shape[1], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    biases2 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)
    # ys_pre = tf.squeeze(tf.matmul(tf.concat(1, [h1, ff]), weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_2)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_1)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    # var_list1 = [p, q, weights1_2]
    # var_list1 = [p, q]
    # var_list2 = [weights1_1, weights2, biases1, biases2]
    # op1 = tf.train.AdamOptimizer(params['learning_rate1']).minimize(loss, var_list = var_list1)
    # op2 = tf.train.AdamOptimizer(params['learning_rate2']).minimize(loss, var_list = var_list2)
    # train_step = tf.group(op1, op2)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        np.random.shuffle(dtrain)
        data_size = dtrain.shape[0]
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#input layer: bucket_ff, concat([bucket1 * weight1 * bucket1, ...])
def exp9(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    t0 = tf.one_hot(feature_list[:, 0], params['bucket_dims'][0])
    t1 = tf.one_hot(feature_list[:, 1], params['bucket_dims'][1])
    t2 = tf.one_hot(feature_list[:, 2], params['bucket_dims'][2])
    t3 = tf.one_hot(feature_list[:, 3], params['bucket_dims'][3])
    t4 = tf.one_hot(feature_list[:, 4], params['bucket_dims'][4])
    t5 = tf.one_hot(feature_list[:, 5], params['bucket_dims'][5])

    w0 = tf.Variable(tf.random_normal([params['bucket_dims'][0], params['bucket_dims'][0]], 0.0, 0.1))
    w1 = tf.Variable(tf.random_normal([params['bucket_dims'][1], params['bucket_dims'][1]], 0.0, 0.1))
    w2 = tf.Variable(tf.random_normal([params['bucket_dims'][2], params['bucket_dims'][2]], 0.0, 0.1))
    w3 = tf.Variable(tf.random_normal([params['bucket_dims'][3], params['bucket_dims'][3]], 0.0, 0.1))
    w4 = tf.Variable(tf.random_normal([params['bucket_dims'][4], params['bucket_dims'][4]], 0.0, 0.1))
    w5 = tf.Variable(tf.random_normal([params['bucket_dims'][5], params['bucket_dims'][5]], 0.0, 0.1))

    f0 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t0, u1s), w0) * tf.nn.embedding_lookup(t0, u2s), 1)
    f1 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t1, u1s), w1) * tf.nn.embedding_lookup(t1, u2s), 1)
    f2 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t2, u1s), w2) * tf.nn.embedding_lookup(t2, u2s), 1)
    f3 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t3, u1s), w3) * tf.nn.embedding_lookup(t3, u2s), 1)
    f4 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t4, u1s), w4) * tf.nn.embedding_lookup(t4, u2s), 1)
    f5 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t5, u1s), w5) * tf.nn.embedding_lookup(t5, u2s), 1)
    ff = tf.transpose([f0, f1, f2, f3, f4, f5])

    weights1_2 = tf.Variable(tf.random_uniform([len(params['bucket_dims']), params['h1_size']], -0.1, 0.1))#(tf.zeros([feature_list.shape[1] * 2, params['h1_size']]))#(tf.random_uniform([feature_list.shape[1] * 2, params['h1_size']], 0, 0))
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(ff, weights1_2) + biases1
    # h1 = tf.matmul(dot_e, weights1_1) + biases1

    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.01, 0.01))
    # biases2 = tf.Variable(tf.zeros([params['h2_size']]))
    # h2 = tf.matmul(h1, weights2) + biases2
    #
    # weights3 = tf.Variable(tf.random_uniform([params['h2_size'], 1], -0.01, 0.01))
    # biases3 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h2, weights3) + biases3)

    weights2 = tf.Variable(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'] + 2 * feature_list.shape[1], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    biases2 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)
    # ys_pre = tf.squeeze(tf.matmul(tf.concat(1, [h1, ff]), weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_2)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_1)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    # var_list1 = [p, q, weights1_2]
    # var_list1 = [p, q]
    # var_list2 = [weights1_1, weights2, biases1, biases2]
    # op1 = tf.train.AdamOptimizer(params['learning_rate1']).minimize(loss, var_list = var_list1)
    # op2 = tf.train.AdamOptimizer(params['learning_rate2']).minimize(loss, var_list = var_list2)
    # train_step = tf.group(op1, op2)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - tf.sigmoid(ys_pre))))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_pre), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        np.random.shuffle(dtrain)
        data_size = dtrain.shape[0]
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#input layer: concat(dot_e, bucket_ff), concat([bucket1 * weight1 * bucket1, ...])
def exp10(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    t0 = tf.one_hot(feature_list[:, 0], params['bucket_dims'][0])
    t1 = tf.one_hot(feature_list[:, 1], params['bucket_dims'][1])
    t2 = tf.one_hot(feature_list[:, 2], params['bucket_dims'][2])
    t3 = tf.one_hot(feature_list[:, 3], params['bucket_dims'][3])
    t4 = tf.one_hot(feature_list[:, 4], params['bucket_dims'][4])
    t5 = tf.one_hot(feature_list[:, 5], params['bucket_dims'][5])

    w0 = tf.Variable(tf.random_normal([params['bucket_dims'][0], params['bucket_dims'][0]], 0.0, 0.1))
    w1 = tf.Variable(tf.random_normal([params['bucket_dims'][1], params['bucket_dims'][1]], 0.0, 0.1))
    w2 = tf.Variable(tf.random_normal([params['bucket_dims'][2], params['bucket_dims'][2]], 0.0, 0.1))
    w3 = tf.Variable(tf.random_normal([params['bucket_dims'][3], params['bucket_dims'][3]], 0.0, 0.1))
    w4 = tf.Variable(tf.random_normal([params['bucket_dims'][4], params['bucket_dims'][4]], 0.0, 0.1))
    w5 = tf.Variable(tf.random_normal([params['bucket_dims'][5], params['bucket_dims'][5]], 0.0, 0.1))

    f0 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t0, u1s), w0) * tf.nn.embedding_lookup(t0, u2s), 1)
    f1 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t1, u1s), w1) * tf.nn.embedding_lookup(t1, u2s), 1)
    f2 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t2, u1s), w2) * tf.nn.embedding_lookup(t2, u2s), 1)
    f3 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t3, u1s), w3) * tf.nn.embedding_lookup(t3, u2s), 1)
    f4 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t4, u1s), w4) * tf.nn.embedding_lookup(t4, u2s), 1)
    f5 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t5, u1s), w5) * tf.nn.embedding_lookup(t5, u2s), 1)
    ff = tf.transpose([f0, f1, f2, f3, f4, f5])

    weights1 = tf.Variable(tf.random_uniform([len(params['bucket_dims']) + params['embedding_size'], params['h1_size']], -0.1, 0.1))#(tf.zeros([feature_list.shape[1] * 2, params['h1_size']]))#(tf.random_uniform([feature_list.shape[1] * 2, params['h1_size']], 0, 0))
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(tf.concat(1, [ff, dot_e]), weights1) + biases1
    # h1 = tf.matmul(dot_e, weights1_1) + biases1

    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.01, 0.01))
    # biases2 = tf.Variable(tf.zeros([params['h2_size']]))
    # h2 = tf.matmul(h1, weights2) + biases2
    #
    # weights3 = tf.Variable(tf.random_uniform([params['h2_size'], 1], -0.01, 0.01))
    # biases3 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h2, weights3) + biases3)

    weights2 = tf.Variable(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'] + 2 * feature_list.shape[1], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    biases2 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)
    # ys_pre = tf.squeeze(tf.matmul(tf.concat(1, [h1, ff]), weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_1)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))

    # train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    var_list1 = [p, q]
    var_list2 = [weights1, weights2, biases1, biases2, w0, w1, w2, w3, w4, w5]
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
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#input layer: dot_e, concat(h1, bucket_ff), concat([bucket1 * weight1 * bucket1, ...])
def exp11(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    t0 = tf.one_hot(feature_list[:, 0], params['bucket_dims'][0])
    t1 = tf.one_hot(feature_list[:, 1], params['bucket_dims'][1])
    t2 = tf.one_hot(feature_list[:, 2], params['bucket_dims'][2])
    t3 = tf.one_hot(feature_list[:, 3], params['bucket_dims'][3])
    t4 = tf.one_hot(feature_list[:, 4], params['bucket_dims'][4])
    t5 = tf.one_hot(feature_list[:, 5], params['bucket_dims'][5])

    w0 = tf.Variable(tf.random_normal([params['bucket_dims'][0], params['bucket_dims'][0]], 0.0, 0.1))
    w1 = tf.Variable(tf.random_normal([params['bucket_dims'][1], params['bucket_dims'][1]], 0.0, 0.1))
    w2 = tf.Variable(tf.random_normal([params['bucket_dims'][2], params['bucket_dims'][2]], 0.0, 0.1))
    w3 = tf.Variable(tf.random_normal([params['bucket_dims'][3], params['bucket_dims'][3]], 0.0, 0.1))
    w4 = tf.Variable(tf.random_normal([params['bucket_dims'][4], params['bucket_dims'][4]], 0.0, 0.1))
    w5 = tf.Variable(tf.random_normal([params['bucket_dims'][5], params['bucket_dims'][5]], 0.0, 0.1))

    f0 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t0, u1s), w0) * tf.nn.embedding_lookup(t0, u2s), 1)
    f1 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t1, u1s), w1) * tf.nn.embedding_lookup(t1, u2s), 1)
    f2 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t2, u1s), w2) * tf.nn.embedding_lookup(t2, u2s), 1)
    f3 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t3, u1s), w3) * tf.nn.embedding_lookup(t3, u2s), 1)
    f4 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t4, u1s), w4) * tf.nn.embedding_lookup(t4, u2s), 1)
    f5 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t5, u1s), w5) * tf.nn.embedding_lookup(t5, u2s), 1)
    ff = tf.transpose([f0, f1, f2, f3, f4, f5])

    weights1 = tf.Variable(tf.random_uniform([params['embedding_size'], params['h1_size']], -0.1, 0.1))#(tf.zeros([feature_list.shape[1] * 2, params['h1_size']]))#(tf.random_uniform([feature_list.shape[1] * 2, params['h1_size']], 0, 0))
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(dot_e, weights1) + biases1
    # h1 = tf.matmul(dot_e, weights1_1) + biases1

    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.01, 0.01))
    # biases2 = tf.Variable(tf.zeros([params['h2_size']]))
    # h2 = tf.matmul(h1, weights2) + biases2
    #
    # weights3 = tf.Variable(tf.random_uniform([params['h2_size'], 1], -0.01, 0.01))
    # biases3 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h2, weights3) + biases3)

    weights2 = tf.Variable(tf.random_uniform([len(params['bucket_dims']) + params['h1_size'], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'] + 2 * feature_list.shape[1], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    biases2 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(tf.concat(1, [ff, h1]), weights2) + biases2)
    # ys_pre = tf.squeeze(tf.matmul(tf.concat(1, [h1, ff]), weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_1)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))

    # train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    var_list1 = [p, q]
    var_list2 = [weights1, weights2, biases1, biases2, w0, w1, w2, w3, w4, w5]
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
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#input layer: dot_e, concat(h1, bucket_ff), concat([bucket1 * weight1 * bucket1, ...]), 2 hidden layer
def exp12(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    dot_e = e1 * e2

    t0 = tf.one_hot(feature_list[:, 0], params['bucket_dims'][0])
    t1 = tf.one_hot(feature_list[:, 1], params['bucket_dims'][1])
    t2 = tf.one_hot(feature_list[:, 2], params['bucket_dims'][2])
    t3 = tf.one_hot(feature_list[:, 3], params['bucket_dims'][3])
    t4 = tf.one_hot(feature_list[:, 4], params['bucket_dims'][4])
    t5 = tf.one_hot(feature_list[:, 5], params['bucket_dims'][5])

    w0 = tf.Variable(tf.random_normal([params['bucket_dims'][0], params['bucket_dims'][0]], 0.0, 0.1))
    w1 = tf.Variable(tf.random_normal([params['bucket_dims'][1], params['bucket_dims'][1]], 0.0, 0.1))
    w2 = tf.Variable(tf.random_normal([params['bucket_dims'][2], params['bucket_dims'][2]], 0.0, 0.1))
    w3 = tf.Variable(tf.random_normal([params['bucket_dims'][3], params['bucket_dims'][3]], 0.0, 0.1))
    w4 = tf.Variable(tf.random_normal([params['bucket_dims'][4], params['bucket_dims'][4]], 0.0, 0.1))
    w5 = tf.Variable(tf.random_normal([params['bucket_dims'][5], params['bucket_dims'][5]], 0.0, 0.1))

    f0 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t0, u1s), w0) * tf.nn.embedding_lookup(t0, u2s), 1)
    f1 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t1, u1s), w1) * tf.nn.embedding_lookup(t1, u2s), 1)
    f2 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t2, u1s), w2) * tf.nn.embedding_lookup(t2, u2s), 1)
    f3 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t3, u1s), w3) * tf.nn.embedding_lookup(t3, u2s), 1)
    f4 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t4, u1s), w4) * tf.nn.embedding_lookup(t4, u2s), 1)
    f5 = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(t5, u1s), w5) * tf.nn.embedding_lookup(t5, u2s), 1)
    ff = tf.transpose([f0, f1, f2, f3, f4, f5])

    weights1 = tf.Variable(tf.random_uniform([params['embedding_size'], params['h1_size']], -0.1, 0.1))
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(dot_e, weights1) + biases1

    weights2 = tf.Variable(tf.random_uniform([len(params['bucket_dims']) + params['h1_size'], params['h2_size']], -0.1, 0.1))
    biases2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.matmul(tf.concat(1, [ff, h1]), weights2) + biases2

    weights3 = tf.Variable(tf.random_uniform([params['h2_size'], 1], -0.1, 0.1))
    biases3 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h2, weights3) + biases3)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))

    # train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    var_list1 = [p, q]
    var_list2 = [weights1, biases1, weights2, biases2, weights3, biases3, w0, w1, w2, w3, w4, w5]
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
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#input layer: e1 * w * e2, 2 hidden layer
def exp13(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))

    features = tf.constant(feature_list)

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    weights0 = tf.Variable(tf.random_uniform([params['embedding_size'], params['embedding_size']], -0.1, 0.1))
    biases0 = tf.Variable(tf.zeros([params['embedding_size']]))
    ee = tf.matmul(e1, weights0) * e2 + biases0

    weights1_1 = tf.Variable(tf.random_uniform([params['embedding_size'], params['h1_size']], -0.1, 0.1))#(tf.zeros([params['embedding_size'], params['h1_size']]))#
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(ee, weights1_1) + biases1

    weights2 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.1, 0.1))
    biases2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.matmul(h1, weights2) + biases2

    # weights3 = tf.Variable(tf.random_uniform([params['h2_size'], params['h3_size']], -0.1, 0.1))
    # biases3 = tf.Variable(tf.zeros([params['h3_size']]))
    # h3 = tf.matmul(h2, weights3) + biases3

    weights3 = tf.Variable(tf.random_uniform([params['h2_size'], 1], -0.1, 0.1))
    biases3 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h2, weights3) + biases3)

    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    # biases2 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_1)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))

    # train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    var_list1 = [p, q]
    var_list2 = [weights1_1, weights2, biases1, biases2]
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
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                tmp, rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([weights2, rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#input layer: concat(e1, e2), 2 hidden layer
def exp14(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))

    features = tf.constant(feature_list)

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)
    ee = tf.concat(1, [e1, e2])

    weights1_1 = tf.Variable(tf.random_uniform([params['embedding_size'] * 2, params['h1_size']], -0.1, 0.1))#(tf.zeros([params['embedding_size'], params['h1_size']]))#
    biases1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(ee, weights1_1) + biases1

    weights2 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.1, 0.1))
    biases2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.matmul(h1, weights2) + biases2

    # weights3 = tf.Variable(tf.random_uniform([params['h2_size'], params['h3_size']], -0.1, 0.1))
    # biases3 = tf.Variable(tf.zeros([params['h3_size']]))
    # h3 = tf.matmul(h2, weights3) + biases3

    weights3 = tf.Variable(tf.random_uniform([params['h2_size'], 1], -0.1, 0.1))
    biases3 = tf.Variable(tf.zeros([1]))
    ys_pre = tf.squeeze(tf.matmul(h2, weights3) + biases3)

    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    # biases2 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_1)) + tf.reduce_sum(tf.square(weights2)) + tf.reduce_sum(tf.square(biases1)) + tf.reduce_sum(tf.square(biases2)))

    # train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    var_list1 = [p, q]
    var_list2 = [weights1_1, weights2, biases1, biases2]
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
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                tmp, rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([weights2, rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#output layer: e1 > h1 > h2 > h3 * w * e1 > h1 > h2 > h3
def exp15(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)

    weights1_1 = tf.Variable(tf.random_uniform([params['embedding_size'], params['h1_size']], -0.1, 0.1))
    biases1_1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1_1 = tf.matmul(e1, weights1_1) + biases1_1

    weights2_1 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.1, 0.1))
    biases2_1 = tf.Variable(tf.zeros([params['h2_size']]))
    h2_1 = tf.matmul(h1_1, weights2_1) + biases2_1

    weights3_1 = tf.Variable(tf.random_uniform([params['h2_size'], params['h3_size']], -0.1, 0.1))
    biases3_1 = tf.Variable(tf.zeros([params['h3_size']]))
    h3_1 = tf.matmul(h2_1, weights3_1) + biases3_1

    weights1_2 = tf.Variable(tf.random_uniform([params['embedding_size'], params['h1_size']], -0.1, 0.1))
    biases1_2 = tf.Variable(tf.zeros([params['h1_size']]))
    h1_2 = tf.matmul(e2, weights1_2) + biases1_2

    weights2_2 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.1, 0.1))
    biases2_2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2_2 = tf.matmul(h1_2, weights2_2) + biases2_2

    weights3_2 = tf.Variable(tf.random_uniform([params['h2_size'], params['h3_size']], -0.1, 0.1))
    biases3_2 = tf.Variable(tf.zeros([params['h3_size']]))
    h3_2 = tf.matmul(h2_2, weights3_2) + biases3_2

    # weights3 = tf.Variable(tf.random_uniform([params['h2_size'], params['h3_size']], -0.1, 0.1))
    # biases3 = tf.Variable(tf.zeros([params['h3_size']]))
    # h3 = tf.matmul(h2, weights3) + biases3

    weights4 = tf.Variable(tf.random_uniform([params['h3_size'], params['h3_size']], -0.1, 0.1))
    biases4 = tf.Variable(tf.zeros([1]))

    ys_pre = tf.reduce_sum(tf.matmul(h3_1, weights4) * h3_2, 1) + biases4#tf.reduce_sum(h3_1 * h3_2, 1)#

    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    # biases2 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_1)) + tf.reduce_sum(tf.square(weights1_2)) + tf.reduce_sum(tf.square(weights2_1)) + tf.reduce_sum(tf.square(weights2_2)) + tf.reduce_sum(tf.square(weights3_1)) + tf.reduce_sum(tf.square(weights3_2)) + tf.reduce_sum(tf.square(weights4)) + tf.reduce_sum(tf.square(biases1_1)) + tf.reduce_sum(tf.square(biases1_2)) + tf.reduce_sum(tf.square(biases2_1)) + tf.reduce_sum(tf.square(biases2_2)) + tf.reduce_sum(tf.square(biases3_1)) + tf.reduce_sum(tf.square(biases3_2)) + tf.reduce_sum(tf.square(biases4)))

    # train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    var_list1 = [p, q]
    var_list2 = [weights1_1, weights1_2, weights2_1, weights2_2, weights3_1, weights3_2, weights4, biases1_1, biases1_2, biases2_1, biases2_2, biases3_1, biases3_2, biases4]
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
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                weights4_v, p_v, rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([tf.reduce_mean(weights4), tf.reduce_mean(p), rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print weights4_v, p_v
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#output layer: (e1 > h1 > h2 > h3 + f1 > h1 > h2 > h3) * w * (e2 > h1 > h2 > h3 + e2 > h1 > h2 > h3)
def exp16(params):
    feature_list = np.float32(load_dict(params['user_feature_list_file_path']))

    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')
    dtrain -= [1, 1, 0]
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')
    dtest -= [1, 1, 0]

    p = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))
    q = tf.Variable(tf.random_uniform([params['node_num'], params['embedding_size']], -0.1, 0.1))

    u1s = tf.placeholder(tf.int32, shape=[None])
    u2s = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    t0 = tf.one_hot(feature_list[:, 0], params['bucket_dims'][0])
    t1 = tf.one_hot(feature_list[:, 1], params['bucket_dims'][1])
    t2 = tf.one_hot(feature_list[:, 2], params['bucket_dims'][2])
    t3 = tf.one_hot(feature_list[:, 3], params['bucket_dims'][3])
    t4 = tf.one_hot(feature_list[:, 4], params['bucket_dims'][4])
    t5 = tf.one_hot(feature_list[:, 5], params['bucket_dims'][5])
    features = tf.concat(1, [t0, t1, t2, t3, t4, t5])

    e1 = tf.nn.embedding_lookup(p, u1s)
    e2 = tf.nn.embedding_lookup(q, u2s)

    f1 = tf.nn.embedding_lookup(features, u1s)
    f2 = tf.nn.embedding_lookup(features, u2s)

    weights1_1 = tf.Variable(tf.random_uniform([params['embedding_size'], params['h1_size']], -0.1, 0.1))
    biases1_1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1_1 = tf.matmul(e1, weights1_1) + biases1_1

    weights2_1 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.1, 0.1))
    biases2_1 = tf.Variable(tf.zeros([params['h2_size']]))
    h2_1 = tf.matmul(h1_1, weights2_1) + biases2_1

    weights3_1 = tf.Variable(tf.random_uniform([params['h2_size'], params['h3_size']], -0.1, 0.1))
    biases3_1 = tf.Variable(tf.zeros([params['h3_size']]))
    h3_1 = tf.matmul(h2_1, weights3_1) + biases3_1

    weights1_2 = tf.Variable(tf.random_uniform([params['embedding_size'], params['h1_size']], -0.1, 0.1))
    biases1_2 = tf.Variable(tf.zeros([params['h1_size']]))
    h1_2 = tf.matmul(e2, weights1_2) + biases1_2

    weights2_2 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.1, 0.1))
    biases2_2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2_2 = tf.matmul(h1_2, weights2_2) + biases2_2

    weights3_2 = tf.Variable(tf.random_uniform([params['h2_size'], params['h3_size']], -0.1, 0.1))
    biases3_2 = tf.Variable(tf.zeros([params['h3_size']]))
    h3_2 = tf.matmul(h2_2, weights3_2) + biases3_2

    weights1_3 = tf.Variable(tf.random_uniform([np.sum(params['bucket_dims']), params['h1_size']], -0.1, 0.1))
    biases1_3 = tf.Variable(tf.zeros([params['h1_size']]))
    h1_3 = tf.matmul(f1, weights1_3) + biases1_3

    weights2_3 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.1, 0.1))
    biases2_3 = tf.Variable(tf.zeros([params['h2_size']]))
    h2_3 = tf.matmul(h1_3, weights2_3) + biases2_3

    weights3_3 = tf.Variable(tf.random_uniform([params['h2_size'], params['h3_size']], -0.1, 0.1))
    biases3_3 = tf.Variable(tf.zeros([params['h3_size']]))
    h3_3 = tf.matmul(h2_3, weights3_3) + biases3_3

    weights1_4 = tf.Variable(tf.random_uniform([np.sum(params['bucket_dims']), params['h1_size']], -0.1, 0.1))
    biases1_4 = tf.Variable(tf.zeros([params['h1_size']]))
    h1_4 = tf.matmul(f2, weights1_4) + biases1_4

    weights2_4 = tf.Variable(tf.random_uniform([params['h1_size'], params['h2_size']], -0.1, 0.1))
    biases2_4 = tf.Variable(tf.zeros([params['h2_size']]))
    h2_4 = tf.matmul(h1_4, weights2_4) + biases2_4

    weights3_4 = tf.Variable(tf.random_uniform([params['h2_size'], params['h3_size']], -0.1, 0.1))
    biases3_4 = tf.Variable(tf.zeros([params['h3_size']]))
    h3_4 = tf.matmul(h2_4, weights3_4) + biases3_4

    # weights3 = tf.Variable(tf.random_uniform([params['h2_size'], params['h3_size']], -0.1, 0.1))
    # biases3 = tf.Variable(tf.zeros([params['h3_size']]))
    # h3 = tf.matmul(h2, weights3) + biases3

    weights4 = tf.Variable(tf.random_uniform([params['h3_size'], params['h3_size']], -0.1, 0.1))
    biases4 = tf.Variable(tf.zeros([1]))

    ys_pre = tf.reduce_sum(tf.matmul(h3_1 + h3_3, weights4) * (h3_2 + h3_4), 1) + biases4#tf.reduce_sum(h3_1 * h3_2, 1)#

    # weights2 = tf.Variable(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))#(tf.zeros([params['h1_size'], 1]))#(tf.random_uniform([params['h1_size'], 1], -0.1, 0.1))
    # biases2 = tf.Variable(tf.zeros([1]))
    # ys_pre = tf.squeeze(tf.matmul(h1, weights2) + biases2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_pre, ys)) + params['beta'] * (tf.reduce_sum(tf.square(weights1_1)) + tf.reduce_sum(tf.square(weights1_2)) + tf.reduce_sum(tf.square(weights2_1)) + tf.reduce_sum(tf.square(weights2_2)) + tf.reduce_sum(tf.square(weights3_1)) + tf.reduce_sum(tf.square(weights3_2)) + tf.reduce_sum(tf.square(weights4)) + tf.reduce_sum(tf.square(biases1_1)) + tf.reduce_sum(tf.square(biases1_2)) + tf.reduce_sum(tf.square(biases2_1)) + tf.reduce_sum(tf.square(biases2_2)) + tf.reduce_sum(tf.square(biases3_1)) + tf.reduce_sum(tf.square(biases3_2)) + tf.reduce_sum(tf.square(biases4)))

    # train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss)
    var_list1 = [p, q]
    var_list2 = [weights1_1, weights1_2, weights2_1, weights2_2, weights3_1, weights3_2, weights4, biases1_1, biases1_2, biases2_1, biases2_2, biases3_1, biases3_2, biases4]
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
        print 'training...'
        for i in range(params['step']):
            start = params['batch_size'] * i % data_size
            end = params['batch_size'] * (i + 1) % data_size
            if end <= start:
                continue
            if i % 100 == 0:
                weights4_v, p_v, rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([tf.reduce_mean(weights4), tf.reduce_mean(p), rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                t_rmse_v, t_loss_v, t_y_trues, t_y_pres, t_accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
                print weights4_v, p_v
                print '----step%d: rmse: %f, loss: %f, auc: %f, accuracy: %f ----test: rmse: %f, loss: %f, auc: %f, accuracy: %f' % (i, rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v, t_rmse_v, t_loss_v, get_auc(t_y_trues, t_y_pres), t_accuracy_v)
            feed_dict = {u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])}
            sess.run(train_step, feed_dict)
            # _, loss_v, y_pres, weights2_v = sess.run([train_step, loss, tf.sigmoid(ys_pre), weights2], feed_dict)
            # print loss_v,weights2_v
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        print 'train result -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        print 'testing...'
        rmse_v, loss_v, y_trues, y_pres, accuracy_v = sess.run([rmse, loss, ys, tf.sigmoid(ys_pre), accuracy], feed_dict={u1s: dtest[:, 0], u2s: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print 'test result  -> rmse: %f, loss: %f, auc: %f, accuracy: %f' % (rmse_v, loss_v, get_auc(y_trues, y_pres), accuracy_v)
        # for i in range(y_trues.shape[0]):
        #     if y_trues[i] == 1.0:
        #         print y_trues[i], y_pres[i]

#pnn
def exp17(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')#0_based
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')

    # dtrain -= [1, 1, 0]
    # dtest -= [1, 1, 0]

    features = tf.placeholder(tf.int32, shape=[None, len(params['feature_depth']) * 2])
    ys = tf.placeholder(tf.float32, shape=[None])

    feature_list = tf.unpack(features, axis=1)
    feature_num = len(feature_list)
    embedding_list = []
    for i in range(feature_num/2):
        embedding_list.append(tf.Variable(tf.random_normal([params['feature_depth'][i], params['embedding_size']], mean=0.0, stddev=0.1)))

    # w0_list = []
    z = tf.concat(1, [tf.nn.embedding_lookup(embedding_list[i/2], feature_list[i]) for i in range(feature_num)])
    p_list = []
    for i in range(feature_num):
        for j in range(feature_num - i -1):
            p_list.append(tf.reduce_sum(tf.nn.embedding_lookup(embedding_list[i/2], feature_list[i]) * tf.nn.embedding_lookup(embedding_list[(i+j+1)/2], feature_list[i+j+1]), 1, keep_dims=True))
    p = tf.concat(1, p_list)

    wlz = tf.Variable(tf.random_normal([params['embedding_size'] * feature_num, params['h1_size']], mean=0.0, stddev=0.1))
    blz = tf.Variable(tf.zeros([params['h1_size']]))
    lz = tf.matmul(z, wlz) + blz

    wlp = tf.Variable(tf.random_normal([feature_num * (feature_num - 1) / 2, params['h1_size']], mean=0.0, stddev=0.1))
    blp = tf.Variable(tf.zeros([params['h1_size']]))
    lp = tf.matmul(p, wlp) + blp

    b1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = lz + lp + b1

    w2 = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0.0, stddev=0.1))
    b2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.matmul(h1, w2) + b2

    w3 = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0.0, stddev=0.1))
    b3 = tf.Variable(tf.zeros([params['h3_size']]))
    h3 = tf.matmul(h2, w3) + b3

    w4 = tf.Variable(tf.random_normal([params['h3_size'], 1], mean=0.0, stddev=0.1))
    b4 = tf.Variable(tf.zeros([1]))
    ys_ = tf.squeeze(tf.matmul(h3, w4) + b4)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_, ys))

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dtrain_size = dtrain.shape[0]
        for i in range(params['step']):
            start = i * params['batch_size'] % dtrain_size
            end = (i+1) * params['batch_size'] % dtrain_size
            if end <= start:
                feed_dict = {features: dtrain[:, 0:feature_num], ys: dtrain[:, feature_num]}
                loss_v, accuracy_v, y_pres, y_trues = sess.run([loss, accuracy, tf.sigmoid(ys_), ys], feed_dict)
                print 'step%d-> loss: %f, auc: %f, accuracy: %f' % (i, loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)

                feed_dict = {features: dtest[:, 0:feature_num], ys: dtest[:, feature_num]}
                loss_v, accuracy_v, y_pres, y_trues = sess.run([loss, accuracy, tf.sigmoid(ys_), ys], feed_dict)
                print '--------test-> loss: %f, auc: %f, accuracy: %f' % (loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)

                continue

            feed_dict = {features: dtrain[start:end, 0:feature_num], ys: dtrain[start:end, feature_num]}
            sess.run(train_step, feed_dict)

#ppnn, change the inner interaction, delete z
def exp18(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')#0_based
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')

    # dtrain -= [1, 1, 0]
    # dtest -= [1, 1, 0]

    features = tf.placeholder(tf.int32, shape=[None, len(params['feature_depth']) * 2])
    ys = tf.placeholder(tf.float32, shape=[None])

    feature_list = tf.unpack(features, axis=1)
    feature_num = len(feature_list)
    embedding_list = []
    for i in range(feature_num/2):
        embedding_list.append(tf.Variable(tf.random_normal([params['feature_depth'][i], params['embedding_size']], mean=0.0, stddev=0.1)))
    z = tf.concat(1, [tf.nn.embedding_lookup(embedding_list[i/2], feature_list[i]) for i in range(feature_num)])
    p_list = []
    wp_list = []
    for i in range(feature_num):
        wp_list.append([])
        for j in range(feature_num - i -1):
            # wp_list[i].append(tf.Variable(tf.random_normal([params['embedding_size']], mean=0.0, stddev=0.1)))
            # p_list.append(tf.reduce_sum(tf.nn.embedding_lookup(embedding_list[i/2], feature_list[i]) * wp_list[i][j] * tf.nn.embedding_lookup(embedding_list[(i+j+1)/2], feature_list[i+j+1]), 1, keep_dims=True))

            # wp_list[i].append(tf.Variable(tf.random_normal([params['embedding_size']], mean=0.0, stddev=0.1)))
            # p_list.append(tf.reduce_sum(tf.nn.embedding_lookup(embedding_list[i / 2], feature_list[i]) * tf.nn.embedding_lookup(embedding_list[(i + j + 1) / 2], feature_list[i + j + 1]), 1, keep_dims=True))

            wp_list[i].append(tf.Variable(tf.random_normal([params['embedding_size'], params['embedding_size']], mean=0.0, stddev=0.1)))
            p_list.append(tf.reduce_sum(tf.matmul(tf.nn.softmax(tf.nn.embedding_lookup(embedding_list[i/2], feature_list[i])), tf.sigmoid(wp_list[i][j])) * tf.nn.softmax(tf.nn.embedding_lookup(embedding_list[(i+j+1)/2], feature_list[i+j+1])), 1, keep_dims=True))
    p = tf.concat(1, p_list)

    wlz = tf.Variable(tf.random_normal([params['embedding_size'] * feature_num, params['h1_size']], mean=0.0, stddev=0.1))
    blz = tf.Variable(tf.zeros([params['h1_size']]))
    lz = tf.matmul(z, wlz) + blz

    wlp = tf.Variable(tf.random_normal([feature_num * (feature_num - 1) / 2, params['h1_size']], mean=0.0, stddev=0.1))
    blp = tf.Variable(tf.zeros([params['h1_size']]))
    lp = tf.matmul(p, wlp) + blp

    b1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = lp + b1#lz + lp + b1

    w2 = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0.0, stddev=0.1))
    b2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.matmul(h1, w2) + b2

    w3 = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0.0, stddev=0.1))
    b3 = tf.Variable(tf.zeros([params['h3_size']]))
    h3 = tf.matmul(h2, w3) + b3

    w4 = tf.Variable(tf.random_normal([params['h3_size'], 1], mean=0.0, stddev=0.1))
    b4 = tf.Variable(tf.zeros([1]))
    ys_ = tf.squeeze(tf.matmul(h3, w4) + b4)

    wlist = [tf.reduce_sum(tf.square(wlp)), tf.reduce_sum(tf.square(w2)), tf.reduce_sum(tf.square(w3)), tf.reduce_sum(tf.square(w4))]

    for i in range(feature_num):
        for j in range(feature_num-1-i):
            wlist.append(tf.reduce_sum(tf.square(wp_list[i][j])))

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_, ys)) + params['beta'] * tf.reduce_sum(wlist)

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dtrain_size = dtrain.shape[0]
        for i in range(params['step']):
            start = i * params['batch_size'] % dtrain_size
            end = (i+1) * params['batch_size'] % dtrain_size
            if end <= start:
                feed_dict = {features: dtrain[:, 0:feature_num], ys: dtrain[:, feature_num]}
                loss_v, accuracy_v, y_pres, y_trues = sess.run([loss, accuracy, tf.sigmoid(ys_), ys], feed_dict)
                print 'step%d-> loss: %f, auc: %f, accuracy: %f' % (i, loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)

                feed_dict = {features: dtest[:, 0:feature_num], ys: dtest[:, feature_num]}
                loss_v, accuracy_v, y_pres, y_trues = sess.run([loss, accuracy, tf.sigmoid(ys_), ys], feed_dict)
                print '----------------test-> loss: %f, auc: %f, accuracy: %f' % (loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)

                continue

            feed_dict = {features: dtrain[start:end, 0:feature_num], ys: dtrain[start:end, feature_num]}
            sess.run(train_step, feed_dict)

#ppnn, change the inner interaction, delete z, reduce time complexity
def exp19(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')#0_based
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')

    # dtrain -= [1, 1, 0]
    # dtest -= [1, 1, 0]

    features = tf.placeholder(tf.int32, shape=[None, len(params['feature_depth']) * 2])
    ys = tf.placeholder(tf.float32, shape=[None])

    feature_list = tf.unpack(features, axis=1)
    feature_num = len(feature_list)
    embedding_list = []
    for i in range(feature_num/2):
        embedding_list.append(tf.Variable(tf.random_normal([params['feature_depth'][i], params['embedding_size']], mean=0.0, stddev=0.1)))
    # z = tf.concat(1, [tf.nn.embedding_lookup(embedding_list[i/2], feature_list[i]) for i in range(feature_num)])
    p_list = []
    wp_list = []
    for i in range(feature_num/2):
        wp_list.append(tf.Variable(tf.random_normal([params['embedding_size'], params['embedding_size']], mean=0.0, stddev=0.1)))
        p_list.append(tf.reduce_sum(tf.matmul(tf.nn.softmax(tf.nn.embedding_lookup(embedding_list[i], feature_list[2*i])), tf.sigmoid(wp_list[i])) * tf.nn.softmax(tf.nn.embedding_lookup(embedding_list[i], feature_list[2*i+1])), 1, keep_dims=True))
    p = tf.concat(1, p_list)

    # wlz = tf.Variable(tf.random_normal([params['embedding_size'] * feature_num, params['h1_size']], mean=0.0, stddev=0.1))
    # blz = tf.Variable(tf.zeros([params['h1_size']]))
    # lz = tf.matmul(z, wlz) + blz

    wlp = tf.Variable(tf.random_normal([feature_num / 2, params['h1_size']], mean=0.0, stddev=0.1))
    blp = tf.Variable(tf.zeros([params['h1_size']]))
    lp = tf.matmul(p, wlp) + blp

    b1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = lp + b1#lz + lp + b1

    w2 = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0.0, stddev=0.1))
    b2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.matmul(h1, w2) + b2

    w3 = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0.0, stddev=0.1))
    b3 = tf.Variable(tf.zeros([params['h3_size']]))
    h3 = tf.matmul(h2, w3) + b3

    w4 = tf.Variable(tf.random_normal([params['h3_size'], 1], mean=0.0, stddev=0.1))
    b4 = tf.Variable(tf.zeros([1]))
    ys_ = tf.squeeze(tf.matmul(h3, w4) + b4)

    wlist = [tf.reduce_sum(tf.square(wlp)), tf.reduce_sum(tf.square(w2)), tf.reduce_sum(tf.square(w3)), tf.reduce_sum(tf.square(w4))]

    for i in range(feature_num/2):
        wlist.append(tf.reduce_sum(tf.square(wp_list[i])))

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_, ys)) + params['beta'] * tf.reduce_sum(wlist)

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dtrain_size = dtrain.shape[0]
        for i in range(params['step']):
            start = i * params['batch_size'] % dtrain_size
            end = (i+1) * params['batch_size'] % dtrain_size
            if end <= start:
                feed_dict = {features: dtrain[:, 0:feature_num], ys: dtrain[:, feature_num]}
                loss_v, accuracy_v, y_pres, y_trues = sess.run([loss, accuracy, tf.sigmoid(ys_), ys], feed_dict)
                print 'step%d-> loss: %f, auc: %f, accuracy: %f' % (i, loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)

                feed_dict = {features: dtest[:, 0:feature_num], ys: dtest[:, feature_num]}
                loss_v, accuracy_v, y_pres, y_trues = sess.run([loss, accuracy, tf.sigmoid(ys_), ys], feed_dict)
                print '----------------test-> loss: %f, auc: %f, accuracy: %f' % (loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)

                continue

            feed_dict = {features: dtrain[start:end, 0:feature_num], ys: dtrain[start:end, feature_num]}
            sess.run(train_step, feed_dict)

#ppnn, change the inner interaction, delete z, reduce time complexity, several test
def exp20(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')#0_based
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')

    dtrain -= [1, 1, 0]
    dtest -= [1, 1, 0]

    features = tf.placeholder(tf.int32, shape=[None, len(params['feature_depth']) * 2])
    ys = tf.placeholder(tf.float32, shape=[None])

    feature_list = tf.unpack(features, axis=1)
    feature_num = len(feature_list)
    embedding_list = []
    for i in range(feature_num/2):
        embedding_list.append(tf.Variable(tf.random_normal([params['feature_depth'][i], params['embedding_size']], mean=0.0, stddev=0.1)))
    # z = tf.concat(1, [tf.nn.embedding_lookup(embedding_list[i/2], feature_list[i]) for i in range(feature_num)])
    p_list = []
    wp_list = []
    for i in range(feature_num/2):
        wp_list.append([])
        # for j in range(params['multiplicity']):
        wp_list[i].append(tf.Variable(tf.random_normal([params['embedding_size'], params['embedding_size']], mean=0.0, stddev=0.1)))
        p_list.append(tf.reduce_sum(tf.matmul(tf.nn.softmax(tf.nn.embedding_lookup(embedding_list[i], feature_list[2*i])), tf.nn.softmax(wp_list[i][0])) * tf.nn.softmax(tf.nn.embedding_lookup(embedding_list[i], feature_list[2*i+1])), 1, keep_dims=True))
        # wp_list[i].append(tf.Variable(tf.random_normal([params['embedding_size'], params['embedding_size']], mean=0.0, stddev=0.1)))
        # p_list.append(tf.reduce_sum(tf.matmul(tf.nn.softmax(tf.nn.embedding_lookup(embedding_list[i], feature_list[2*i+1])), tf.nn.softmax(wp_list[i][1])) * tf.nn.softmax(tf.nn.embedding_lookup(embedding_list[i], feature_list[2*i])), 1, keep_dims=True))

    p = tf.concat(1, p_list)

    # wlz = tf.Variable(tf.random_normal([params['embedding_size'] * feature_num, params['h1_size']], mean=0.0, stddev=0.1))
    # blz = tf.Variable(tf.zeros([params['h1_size']]))
    # lz = tf.matmul(z, wlz) + blz

    wlp = tf.Variable(tf.random_normal([feature_num / 2, params['h1_size']], mean=0.0, stddev=0.1))
    blp = tf.Variable(tf.zeros([params['h1_size']]))
    lp = tf.matmul(p, wlp) + blp

    b1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = lp + b1#lz + lp + b1

    w2 = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0.0, stddev=0.1))
    b2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.matmul(h1, w2) + b2

    w3 = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0.0, stddev=0.1))
    b3 = tf.Variable(tf.zeros([params['h3_size']]))
    h3 = tf.matmul(h2, w3) + b3

    w4 = tf.Variable(tf.random_normal([params['h3_size'], 1], mean=0.0, stddev=0.1))
    b4 = tf.Variable(tf.zeros([1]))
    ys_ = tf.squeeze(tf.matmul(h3, w4) + b4)

    wlist = [tf.reduce_sum(tf.square(wlp)), tf.reduce_sum(tf.square(w2)), tf.reduce_sum(tf.square(w3)), tf.reduce_sum(tf.square(w4))]

    for i in range(feature_num/2):
        wlist.append(tf.reduce_sum(tf.square(wp_list[i])))

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_, ys)) + params['beta'] * tf.reduce_sum(wlist)

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dtrain_size = dtrain.shape[0]
        for i in range(params['step']):
            start = i * params['batch_size'] % dtrain_size
            end = (i+1) * params['batch_size'] % dtrain_size
            if end <= start:
                feed_dict = {features: dtrain[:, 0:feature_num], ys: dtrain[:, feature_num]}
                loss_v, accuracy_v, y_pres, y_trues = sess.run([loss, accuracy, tf.sigmoid(ys_), ys], feed_dict)
                print 'step%d-> loss: %f, auc: %f, accuracy: %f' % (i, loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)

                feed_dict = {features: dtest[:, 0:feature_num], ys: dtest[:, feature_num]}
                loss_v, accuracy_v, y_pres, y_trues = sess.run([loss, accuracy, tf.sigmoid(ys_), ys], feed_dict)
                print '----------------test-> loss: %f, auc: %f, accuracy: %f' % (loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)

                continue

            feed_dict = {features: dtrain[start:end, 0:feature_num], ys: dtrain[start:end, feature_num]}
            sess.run(train_step, feed_dict)

#ppnn, use outer interaction
def exp21(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')  # 0_based
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')

    # dtrain -= [1, 1, 0]
    # dtest -= [1, 1, 0]

    features = tf.placeholder(tf.int32, shape=[None, len(params['feature_depth']) * 2])
    ys = tf.placeholder(tf.float32, shape=[None])

    feature_list = tf.unpack(features, axis=1)
    feature_num = len(feature_list)
    embedding_list = []
    for i in range(feature_num / 2):
        embedding_list.append(tf.Variable(tf.random_normal([params['feature_depth'][i], params['embedding_size']], mean=0.0, stddev=0.1)))
    # z = tf.concat(1, [tf.nn.embedding_lookup(embedding_list[i/2], feature_list[i]) for i in range(feature_num)])
    p_list = []
    for i in range(feature_num / 2):
        p_list.append(tf.concat(1, tf.unpack(tf.batch_matmul(tf.expand_dims(tf.nn.softmax(tf.nn.embedding_lookup(embedding_list[i], feature_list[2*i])), -1), tf.expand_dims(tf.nn.softmax(tf.nn.embedding_lookup(embedding_list[i],feature_list[2*i+1])), -1), adj_y=True), axis=2)))
    p = tf.concat(1, p_list)

    # wlz = tf.Variable(tf.random_normal([params['embedding_size'] * feature_num, params['h1_size']], mean=0.0, stddev=0.1))
    # blz = tf.Variable(tf.zeros([params['h1_size']]))
    # lz = tf.matmul(z, wlz) + blz

    # wlp = tf.Variable(tf.random_normal([feature_num / 2, params['h1_size']], mean=0.0, stddev=0.1))
    # blp = tf.Variable(tf.zeros([params['h1_size']]))
    # lp = tf.matmul(p, wlp) + blp

    w1 = tf.Variable(tf.random_normal([params['embedding_size'] * params['embedding_size'] * feature_num / 2, params['h1_size']], mean=0.0, stddev=0.1))
    b1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = tf.matmul(p, w1) + b1

    w2 = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0.0, stddev=0.1))
    b2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.matmul(h1, w2) + b2

    w3 = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0.0, stddev=0.1))
    b3 = tf.Variable(tf.zeros([params['h3_size']]))
    h3 = tf.matmul(h2, w3) + b3

    w4 = tf.Variable(tf.random_normal([params['h3_size'], 1], mean=0.0, stddev=0.1))
    b4 = tf.Variable(tf.zeros([1]))
    ys_ = tf.squeeze(tf.matmul(h3, w4) + b4)

    wlist = [tf.reduce_sum(tf.square(w1)), tf.reduce_sum(tf.square(w2)), tf.reduce_sum(tf.square(w3)), tf.reduce_sum(tf.square(w4))]

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_, ys)) #+ params['beta'] * tf.reduce_sum(wlist)

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # var_list1 = []
    # for i in range(feature_num / 2):
    #     var_list1.append(embedding_list[i])
    # var_list2 = [w1, b1, w2, b2, w3, b3, w4, b4]
    # op1 = tf.train.AdamOptimizer(params['learning_rate1']).minimize(loss, var_list=var_list1)
    # op2 = tf.train.AdamOptimizer(params['learning_rate2']).minimize(loss, var_list=var_list2)
    # train_step = tf.group(op1, op2)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dtrain_size = dtrain.shape[0]
        for i in range(params['step']):
            start = i * params['batch_size'] % dtrain_size
            end = (i + 1) * params['batch_size'] % dtrain_size
            if end <= start:
                feed_dict = {features: dtrain[:, 0:feature_num], ys: dtrain[:, feature_num]}
                loss_v, accuracy_v, y_pres, y_trues = sess.run([loss, accuracy, tf.sigmoid(ys_), ys], feed_dict)
                print 'step%d-> loss: %f, auc: %f, accuracy: %f' % (
                i, loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)

                feed_dict = {features: dtest[:, 0:feature_num], ys: dtest[:, feature_num]}
                loss_v, accuracy_v, y_pres, y_trues = sess.run([loss, accuracy, tf.sigmoid(ys_), ys], feed_dict)
                print '----------------test-> loss: %f, auc: %f, accuracy: %f' % (
                loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)

                continue

            feed_dict = {features: dtrain[start:end, 0:feature_num], ys: dtrain[start:end, feature_num]}
            sess.run(train_step, feed_dict)

#ppnn, use outer interaction, use + instead of concat
def exp22(params):
    dtrain = np.loadtxt(params['dtrain_file_path'], delimiter=',')  # 0_based
    dtest = np.loadtxt(params['dtest_file_path'], delimiter=',')

    # dtrain -= [1, 1, 0]
    # dtest -= [1, 1, 0]

    features = tf.placeholder(tf.int32, shape=[None, len(params['feature_depth']) * 2])
    ys = tf.placeholder(tf.float32, shape=[None])

    feature_list = tf.unpack(features, axis=1)
    feature_num = len(feature_list)
    embedding_list = []
    for i in range(feature_num / 2):
        embedding_list.append(tf.Variable(tf.random_normal([params['feature_depth'][i], params['embedding_size']], mean=0.0, stddev=0.1)))
    # z = tf.concat(1, [tf.nn.embedding_lookup(embedding_list[i/2], feature_list[i]) for i in range(feature_num)])
    p_list = []
    wlp_list = []
    blp = tf.Variable(tf.zeros([params['h1_size']]))
    lp = blp
    for i in range(feature_num / 2):
        p_list.append(tf.concat(1, tf.unpack(tf.batch_matmul(tf.expand_dims(tf.nn.embedding_lookup(embedding_list[i], feature_list[2*i]), -1), tf.expand_dims(tf.nn.embedding_lookup(embedding_list[i],feature_list[2*i+1]), -1), adj_y=True), axis=2)))
        wlp_list.append(tf.Variable(tf.random_normal([params['embedding_size'] * params['embedding_size'], params['h1_size']], mean=0.0, stddev=0.1)))
        lp  += tf.matmul(p_list[i], wlp_list[i])
    #p = tf.concat(1, p_list)

    # wlz = tf.Variable(tf.random_normal([params['embedding_size'] * feature_num, params['h1_size']], mean=0.0, stddev=0.1))
    # blz = tf.Variable(tf.zeros([params['h1_size']]))
    # lz = tf.matmul(z, wlz) + blz

    # wlp = tf.Variable(tf.random_normal([feature_num / 2, params['h1_size']], mean=0.0, stddev=0.1))
    # blp = tf.Variable(tf.zeros([params['h1_size']]))
    # lp = tf.matmul(p, wlp) + blp

    # w1 = tf.Variable(tf.random_normal([params['embedding_size'] * params['embedding_size'] * feature_num / 2, params['h1_size']], mean=0.0, stddev=0.1))
    # b1 = tf.Variable(tf.zeros([params['h1_size']]))
    h1 = lp#tf.matmul(lp, w1) + b1

    w2 = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean=0.0, stddev=0.1))
    b2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.matmul(h1, w2) + b2

    w3 = tf.Variable(tf.random_normal([params['h2_size'], params['h3_size']], mean=0.0, stddev=0.1))
    b3 = tf.Variable(tf.zeros([params['h3_size']]))
    h3 = tf.matmul(h2, w3) + b3

    w4 = tf.Variable(tf.random_normal([params['h3_size'], 1], mean=0.0, stddev=0.1))
    b4 = tf.Variable(tf.zeros([1]))
    ys_ = tf.squeeze(tf.matmul(h3, w4) + b4)

    # wlist = [tf.reduce_sum(tf.square(w1)), tf.reduce_sum(tf.square(w2)), tf.reduce_sum(tf.square(w3)), tf.reduce_sum(tf.square(w4))]

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_, ys)) #+ params['beta'] * tf.reduce_sum(wlist)

    train_step = tf.train.AdamOptimizer(params['learning_rate']).minimize(loss)
    # var_list1 = []
    # for i in range(feature_num / 2):
    #     var_list1.append(embedding_list[i])
    # var_list2 = [w1, b1, w2, b2, w3, b3, w4, b4]
    # op1 = tf.train.AdamOptimizer(params['learning_rate1']).minimize(loss, var_list=var_list1)
    # op2 = tf.train.AdamOptimizer(params['learning_rate2']).minimize(loss, var_list=var_list2)
    # train_step = tf.group(op1, op2)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys, tf.int32), tf.cast(tf.sigmoid(ys_), tf.int32)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dtrain_size = dtrain.shape[0]
        for i in range(params['step']):
            start = i * params['batch_size'] % dtrain_size
            end = (i + 1) * params['batch_size'] % dtrain_size
            if end <= start:
                feed_dict = {features: dtrain[:, 0:feature_num], ys: dtrain[:, feature_num]}
                loss_v, accuracy_v, y_pres, y_trues = sess.run([loss, accuracy, tf.sigmoid(ys_), ys], feed_dict)
                print 'step%d-> loss: %f, auc: %f, accuracy: %f' % (
                i, loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)

                feed_dict = {features: dtest[:, 0:feature_num], ys: dtest[:, feature_num]}
                loss_v, accuracy_v, y_pres, y_trues = sess.run([loss, accuracy, tf.sigmoid(ys_), ys], feed_dict)
                print '----------------test-> loss: %f, auc: %f, accuracy: %f' % (
                loss_v, roc_auc_score(y_trues, y_pres), accuracy_v)

                continue

            feed_dict = {features: dtrain[start:end, 0:feature_num], ys: dtrain[start:end, feature_num]}
            sess.run(train_step, feed_dict)

def train_model(params):
    print 'gg'

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
    # xgboost('../../data/test/small_train_data_libsvm', '../../data/test/small_test_data_libsvm', {'max_depth':4, 'eta':0.3, 'silent':0, 'objective':'binary:logistic', 'nthread':8}, 15)
    # train_model({'node_num': 10000, 'round': 10, 'learning_rate': 0.02, 'batch_size': 5000, 'lamda': 0.0005, 'learning_rate1': 1,
    #              'learning_rate2': 0.001, 'embedding_size': 4, 'h1_size': 3, 'h2_size':2,
    #             'pq_file_path': '../../data/test/svd_embedding_dict',
    #              'train_data_file_path': '../../data/test/small_train_data_csv', 'model_save_file_path': '../../data/test/model_weights.ckpt'})
    # gen_svd_embeddings('../../data/test/small_data', '../../data/test/svd_embedding_dict')
    # mf_only_with_biases({'dtrain_file_path': '../../data/test/small_train_data_v3_csv', 'dtest_file_path': '../../data/test/small_test_data_v3_csv', 'target_file_path': '../../data/test/mf_dict', 'node_num': 10000, 'embedding_size':4, 'batch_size': 4800, 'step': 5000, 'learning_rate': 0.02, 'beta': 0.0001})
    # mf({'dtrain_file_path': '../../data/test/small_train_data_v3_csv', 'dtest_file_path': '../../data/test/small_test_data_v3_csv', 'target_file_path': '../../data/test/mf_dict', 'node_num': 10000, 'embedding_size':4, 'batch_size': 4800, 'step': 5000, 'learning_rate': 0.02, 'beta': 0.0001})
    # mf_with_weight({'source_file_path': '../../data/test/small_train_data_csv', 'target_file_path': '../../data/test/mf_dict', 'node_num': 10000, 'embedding_size':8, 'h1_size': 5, 'batch_size': 4800, 'step': 3000, 'learning_rate': 0.01, 'beta': 0.0001})
    # mf_with_weight_for_debug({'pq_file_path': '../../data/test/svd_embedding_dict', 'source_file_path': '../../data/test/small_train_data_csv', 'target_file_path': '../../data/test/mf_dict', 'node_num': 10000, 'embedding_size':4, 'h1_size': 3, 'batch_size': 5000, 'step': 10000, 'learning_rate': 0.02, 'beta': 0.0001})
    mf_with_nn({'source_file_path': '../../data/test/small_train_data_v2_csv', 'target_file_path': '../../data/test/mf_dict', 'node_num': 10000, 'embedding_size':8, 'h1_size': 5, 'batch_size': 4800, 'step': 3000, 'learning_rate': 0.01, 'beta': 0.0001})
    # exp1({'dtrain_file_path': '../../data/test/small_train_data_v2_csv', 'dtest_file_path': '../../data/test/small_test_data_v2_csv', 'target_file_path': '../../data/test/mf_dict',
    #      'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'batch_size': 128, 'step': 4000, 'learning_rate': 0.001})
    # exp2({'user_feature_list_file_path': '../../data/test/small_data_user_feature_list',
    #       'dtrain_file_path': '../../data/test/small_train_data_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_csv', 'target_file_path': '../../data/test/mf_dict',
    #       'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'h2_size': 8, 'batch_size': 4800, 'step': 6000,
    #       'learning_rate': 1e-4, 'learning_rate1': 1e-2, 'learning_rate2': 1e-4, 'beta': 1e-9})
    # exp3({'user_feature_list_file_path': '../../data/test/small_data_user_feature_list',
    #       'dtrain_file_path': '../../data/test/small_train_data_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_csv',
    #       'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'h2_size': 0, 'batch_size': 4800, 'step': 2000,
    #       'learning_rate': 3e-3, 'learning_rate1': 0, 'learning_rate2': 0, 'beta': 1e-4})# same learning rate, auc: 0.85
    # exp3({'user_feature_list_file_path': '../../data/test/small_data_user_feature_list',
    #       'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #       'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'h2_size': 0, 'batch_size': 4800, 'step': 6000,
    #       'learning_rate': 0, 'learning_rate1': 1e-2, 'learning_rate2': 3e-7, 'beta': 1e-4})  # different learning rate, auc: 0.93
    # exp4({'user_feature_list_file_path': '../../data/test/small_data_user_feature_list',
    #       'dtrain_file_path': '../../data/test/small_train_data_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_csv',
    #       'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'h2_size': 0, 'batch_size': 4800, 'step': 2500,
    #       'learning_rate': 1e-3, 'learning_rate1': 0, 'learning_rate2': 0, 'beta': 1e-4})# same learning rate, auc: 0.86
    # exp4({'user_feature_list_file_path': '../../data/test/small_data_user_feature_list',
    #       'dtrain_file_path': '../../data/test/small_train_data_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_csv',
    #       'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'h2_size': 0, 'batch_size': 4800, 'step': 6000,
    #       'learning_rate': 0, 'learning_rate1': 1e-2, 'learning_rate2': 3e-7, 'beta': 1e-4})# different learning rate, auc: 0.93
    # exp5({'user_feature_list_file_path': '../../data/test/small_data_user_feature_list',
    #       'dtrain_file_path': '../../data/test/small_train_data_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_csv',
    #       'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'h2_size': 0, 'batch_size': 4800, 'step': 6000,
    #       'learning_rate': 0, 'learning_rate1': 1e-2, 'learning_rate2': 3e-5, 'beta': 1e-4})
    # exp6({'user_feature_list_file_path': '../../data/test/small_data_user_feature_list',
    #       'dtrain_file_path': '../../data/test/small_train_data_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_csv',
    #       'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'h2_size': 0, 'batch_size': 4800, 'step': 6000,
    #       'learning_rate': 0.01, 'learning_rate1': 0, 'learning_rate2': 0, 'beta': 1e-4})
    # exp7({'user_feature_list_file_path': '../../data/test/small_data_user_feature_list',
    #       'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #       'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'h2_size': 8, 'batch_size': 4800, 'step': 5000,
    #       'learning_rate': 0, 'learning_rate1': 3e-2, 'learning_rate2': 3e-6, 'beta': 1e-4})
    # exp8({'user_feature_list_file_path': '../../data/test/small_data_uf_bucket_list',
    #       'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #       'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'batch_size': 4800, 'step': 6000,
    #       'learning_rate': 0.01, 'learning_rate1': 0, 'learning_rate2': 0, 'beta': 1e-4, 'bucket_dims': [2, 50, 2, 30, 100, 100]})
    # exp9({'user_feature_list_file_path': '../../data/test/small_data_uf_bucket_list',
    #       'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #       'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'batch_size': 4800, 'step': 6000,
    #       'learning_rate': 0.01, 'learning_rate1': 0, 'learning_rate2': 0, 'beta': 1e-4, 'bucket_dims': [2, 50, 2, 30, 100, 100]})
    # exp10({'user_feature_list_file_path': '../../data/test/small_data_uf_bucket_list',
    #       'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #       'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'batch_size': 4800, 'step': 6000,
    #       'learning_rate': 0.0, 'learning_rate1': 1e-2, 'learning_rate2': 1e-6, 'beta': 1e-4, 'bucket_dims': [2, 50, 2, 30, 100, 100]})
    # exp11({'user_feature_list_file_path': '../../data/test/small_data_uf_bucket_list',
    #        'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #        'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #        'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'batch_size': 4800, 'step': 6000,
    #        'learning_rate': 0.0, 'learning_rate1': 2e-2, 'learning_rate2': 1e-6, 'beta': 1e-4,
    #        'bucket_dims': [2, 50, 2, 30, 100, 100]})
    # exp12({'user_feature_list_file_path': '../../data/test/small_data_uf_bucket_list',
    #        'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #        'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #        'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'h2_size':8, 'batch_size': 4800, 'step': 10000,
    #        'learning_rate': 0.0, 'learning_rate1': 3e-2, 'learning_rate2': 3e-6, 'beta': 1e-4,
    #        'bucket_dims': [2, 50, 2, 30, 100, 100]})
    # exp13({'user_feature_list_file_path': '../../data/test/small_data_user_feature_list',
    #       'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #       'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'h2_size': 8, 'batch_size': 4800, 'step': 5000,
    #       'learning_rate': 0, 'learning_rate1': 8e-2, 'learning_rate2': 8e-6, 'beta': 1e-4})
    # exp14({'user_feature_list_file_path': '../../data/test/small_data_user_feature_list',
    #       'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #       'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #       'node_num': 10000, 'embedding_size': 20, 'h1_size': 8, 'h2_size': 8, 'batch_size': 4800, 'step': 5000,
    #       'learning_rate': 0, 'learning_rate1': 1e-1, 'learning_rate2': 1e-5, 'beta': 1e-4})
    # exp15({'user_feature_list_file_path': '../../data/test/small_data_user_feature_list',
    #        'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #        'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #        'node_num': 10000, 'embedding_size': 20, 'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'batch_size': 4800, 'step': 5000,
    #        'learning_rate': 0, 'learning_rate1': 5e-1, 'learning_rate2': 5e-5, 'beta': 1e-4})
    # exp16({'user_feature_list_file_path': '../../data/test/small_data_user_feature_list',
    #        'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #        'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #        'node_num': 10000, 'embedding_size': 20, 'h1_size': 20, 'h2_size': 20, 'h3_size': 20, 'batch_size': 4800, 'step': 5000,
    #        'learning_rate': 0, 'learning_rate1': 5e-1, 'learning_rate2': 5e-5, 'beta': 1e-4,
    #        'bucket_dims': [2, 50, 2, 30, 100, 100]})
    # exp17({'dtrain_file_path': '../../data/test/m_train_data_v2_csv', 'dtest_file_path': '../../data/test/m_test_data_v2_csv',
    #        'feature_depth': [100000, 2, 50, 2, 30, 100, 100], 'embedding_size': 20,
    #        'h1_size': 20, 'h2_size': 20, 'h3_size': 20,
    #        'batch_size': 2000, 'step': 30*4000, 'learning_rate': 0.0001})
    # exp18({'dtrain_file_path': '../../data/test/small_train_data_v4_csv',
    #        'dtest_file_path': '../../data/test/small_test_data_v4_csv',
    #        'feature_depth': [10000, 2, 50, 2, 30, 100, 100], 'embedding_size': 10,
    #        'h1_size': 10, 'h2_size': 10, 'h3_size': 10,
    #        'batch_size': 480, 'step': 30000, 'learning_rate': 0.005, 'beta': 1e-6})
    # exp19({'dtrain_file_path': '../../data/test/small_train_data_v4_csv',
    #        'dtest_file_path': '../../data/test/small_test_data_v4_csv',
    #        'feature_depth': [10000, 2, 50, 2, 30, 100, 100], 'embedding_size': 10,
    #        'h1_size': 10, 'h2_size': 10, 'h3_size': 10,
    #        'batch_size': 480, 'step': 60000, 'learning_rate': 0.005, 'beta': 2e-5})
    # exp20({'dtrain_file_path': '../../data/test/small_train_data_v3_csv',
    #        'dtest_file_path': '../../data/test/small_test_data_v3_csv',
    #        'feature_depth': [10000], 'embedding_size': 10,
    #        'multiplicity': 3,
    #        'h1_size': 10, 'h2_size': 10, 'h3_size': 10,
    #        'batch_size': 480, 'step': 30000, 'learning_rate': 0.01, 'beta': 1e-6})
    # exp21({'dtrain_file_path': '../../data/test/small_train_data_v4_csv',
    #        'dtest_file_path': '../../data/test/small_test_data_v4_csv',
    #        'feature_depth': [10000, 2, 50, 2, 30, 100, 100], 'embedding_size': 10,
    #        'multiplicity': 3,
    #        'h1_size': 10, 'h2_size': 10, 'h3_size': 10,
    #        'batch_size': 480, 'step': 30000, 'learning_rate': 0.005, 'learning_rate1': 1e-2, 'learning_rate2': 1e-6, 'beta': 1e-6})
    # exp22({'dtrain_file_path': '../../data/test/small_train_data_v4_csv',
    #        'dtest_file_path': '../../data/test/small_test_data_v4_csv',
    #        'feature_depth': [10000, 2, 50, 2, 30, 100, 100], 'embedding_size': 10,
    #        'multiplicity': 3,
    #        'h1_size': 10, 'h2_size': 10, 'h3_size': 10,
    #        'batch_size': 480, 'step': 30000, 'learning_rate': 0.001, 'learning_rate1': 1e-2, 'learning_rate2': 1e-6,
    #        'beta': 1e-6})

if __name__ == '__main__':
    main()
