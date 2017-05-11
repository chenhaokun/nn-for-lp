import tensorflow as tf
import numpy as np


def saver_save_test():
    a = tf.Variable(tf.ones([3, 2]), name='a')
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver({'a': a})
    with tf.Session() as sess:
        sess.run(init_op)
        a -= 1
        save_path = saver.save(sess, '/model_weights.ckpt')
        print save_path


def saver_restore_test():
    a = tf.Variable(tf.ones([3, 2]), name='a')
    saver = tf.train.Saver({'a': a})
    with tf.Session() as sess:
        saver.restore(sess, '/model_weights.ckpt')
        print sess.run(a)


def get_input_tensors(input_file_path):
    list1 = []
    list2 = []
    list3 = []
    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            items = line[0:len(line) - 1].split(',')
            list1.append(int(items[0]))
            list2.append(int(items[1]))
            list3.append(int(items[2]))
    return tf.convert_to_tensor(list1), tf.convert_to_tensor(list2), tf.convert_to_tensor(list3)


def shuffle_batch_test():
    filenames = ['../../data/test/train_data_csv_test']
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    u1, u2, label = tf.decode_csv(value, record_defaults=[[0], [0], [0.0]])
    u1_batch, u2_batch, label_batch = tf.train.shuffle_batch([u1, u2, label], batch_size=1, capacity=20,
                                                             min_after_dequeue=10)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(coord=coord)
        for i in range(40):
            u1val, u2val, lval = sess.run([u1_batch, u2_batch, label_batch])
            print u1val, u2val, lval
        coord.request_stop()
        coord.join(thread)


def train_model(input_file_path, params, model_save_file_path):
    '''filenames=[input_file_path]
    filename_queue=tf.train.string_input_producer(filenames)
    reader=tf.TextLineReader()
    key,value=reader.read(filename_queue)
    u1,u2,y=tf.decode_csv(value,record_defaults=[[0],[0],[0.0]])'''
    dtrain = np.loadtxt(input_file_path, delimiter=',', dtype=np.int32)
    dtrain -= [1, 1, 0]

    # u1s,u2s,ys=tf.train.shuffle_batch([u1,u2,y],batch_size=params['batch_size'],capacity=3000,min_after_dequeue=1000)

    u1s = tf.placeholder(tf.int32, name='u1s')
    u2s = tf.placeholder(tf.int32, name='u2s')
    ys = tf.placeholder(tf.float32, name='ys')
    # oh_u1s=tf.one_hot(u1s-1,depth=params['node_num'])
    # oh_u2s=tf.one_hot(u2s-1,depth=params['node_num'])

    # u1s -= 1
    # u2s -= 1

    # weights1_1 = tf.Variable(tf.random_normal([params['node_num'], params['h_len']], stddev=1), name='weights1_1')
    # weights1_2 = tf.Variable(tf.random_normal([params['node_num'], params['h_len']], stddev=1), name='weights1_2')
    # e1 = tf.nn.embedding_lookup(weights1_1, u1s)
    # e2 = tf.nn.embedding_lookup(weights1_2, u2s)
    # ys_ = tf.reduce_sum(e1*e2,1)

    weights1_1 = tf.Variable(tf.random_normal([params['node_num'], params['h_len']], mean=0, stddev=1), name='weights1_1')
    weights1_2 = tf.Variable(tf.random_normal([params['node_num'], params['h_len']], mean=0, stddev=1), name='weights1_2')
    # weights1_1 = tf.Variable(tf.zeros([100, 10]))
    # weights1_2 = tf.Variable(tf.zeros([100, 10]))
    biases1 = tf.Variable(tf.zeros([params['h_len'] * 2]), name='biases1')
    e1 = tf.nn.embedding_lookup(weights1_1, u1s)
    e2 = tf.nn.embedding_lookup(weights1_2, u2s)
    h1 = tf.concat(1, [e1, e2]) + biases1

    weights2_1 = tf.Variable(tf.random_normal([params['h_len'] * 2, 1], mean=0, stddev=1), name='weights2_1')
    weights2_2 = tf.Variable(tf.random_normal([params['h_len'], 1], stddev=1), name='weight2_2')
    # weights2 = tf.Variable(tf.zeros([20, 10]))
    biases2 = tf.Variable(tf.ones([1]), name='biases2')
    # h2=tf.sigmoid(tf.matmul(h1,weights2)+biases2)
    ys_ = tf.squeeze(tf.matmul(h1, weights2_1) + biases2)#tf.squeeze(tf.matmul(h1, weights2_1) + tf.matmul(e1*e2, weights2_2) + biases2)

    # weights3=tf.Variable(tf.random_normal([params['h_len'], 1], stddev=0.1),name='weight3')
    # weights3 = tf.Variable(tf.zeros([10, 1]))
    # biases3=tf.Variable(tf.zeros([1]),name='biases3')
    # ys_=tf.squeeze(tf.matmul(h2,weights3)+biases3)

    # weights4 = tf.Variable(tf.random_normal([20, 1]))
    # bias4 = tf.Variable(0.0)
    # ys_ = tf.matmul(h1, weights4) + bias4

    # cross_entropy=-tf.reduce_sum(ys*tf.log(ys_) + (1-ys)*tf.log(1-ys_))
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_, ys))
    # cross_entropy = tf.reduce_mean(tf.sqrt(tf.square(ys_ - ys)))# + 0.0002 * tf.reduce_sum(tf.reduce_sum(tf.square(e1)+tf.square(e2)))

    # train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(cross_entropy)
    var_list1 = [weights1_1, weights1_2]
    var_list2 = [weights2_1, weights2_2, biases1, biases2]
    op1 = tf.train.GradientDescentOptimizer(params['learning_rate1']).minimize(cross_entropy, var_list = var_list1)
    op2 = tf.train.GradientDescentOptimizer(params['learning_rate2']).minimize(cross_entropy, var_list = var_list2)
    train_step = tf.group(op1, op2)

    # saver=tf.train.Saver({'weights1_1':weights1_1,'weights1_2':weights1_2,'biases1':biases1,'weight2':weights2,'biases2':biases2,'weights3':weights3,'biases3':biases3})

    auc, _= tf.contrib.metrics.streaming_auc(tf.sigmoid(ys_), dtrain[:, 2])
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys,tf.int32),tf.cast(tf.sigmoid(ys_),tf.int32)),tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # coord=tf.train.Coordinator()
        # thread=tf.train.start_queue_runners(coord=coord)
        round_count=0
        print 'cross entropy of round%d: %f' % (round_count, sess.run(cross_entropy, feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])}))
        print sess.run(ys_, feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        np.random.shuffle(dtrain)

        for i in range(int(float(params['round']) * dtrain.shape[0] / params['batch_size'])):
            start = i * params['batch_size'] % dtrain.shape[0]
            end = (i + 1) * params['batch_size'] % dtrain.shape[0]
            if end <= start:
                round_count+=1
                auc_v, accuracy_v, ce, y_pres, y_trues, weights1_1_v, weights1_2_v, weights2_1_v = sess.run([auc, accuracy, cross_entropy, tf.round(tf.sigmoid(ys_)), ys, weights1_1, weights1_2, weights2_1], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                print 'round%d: %.3f' % (round_count,ce)
                print weights1_1_v, weights2_1_v
                print auc_v, accuracy_v, y_pres.mean()
                # t1,t2,t3 =  sess.run([tf.cast(ys,tf.int32), tf.cast(ys_,tf.int32), tf.equal(tf.cast(ys,tf.int32),tf.cast(ys_,tf.int32))], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                # for j in range(y_pres.shape[0]):
                #     print y_pres[j], y_trues[j]
                np.random.shuffle(dtrain)
                continue
            sess.run(train_step, feed_dict={u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])})

            # print sess.run(ys_).shape
            # _,  _ys_, ce, weights1_1_mean, weights1_2_mean, weights2_1_mean, weights2_2_mean, biases1_mean, biases2_mean = sess.run([train_step, ys_, cross_entropy, tf.reduce_mean(weights1_1), tf.reduce_mean(weights1_2), tf.reduce_mean(weights2_1), tf.reduce_mean(weights2_2), tf.reduce_mean(biases1), tf.reduce_mean(biases2)], feed_dict={u1s:dtrain[start:end, 0], u2s:dtrain[start:end, 1], ys:np.float32(dtrain[start:end, 2])})
            # print weights1_1_mean, weights1_2_mean, weights2_1_mean, weights2_2_mean, biases1_mean, biases2_mean

        # coord.request_stop()
        # coord.join(thread)
        # saver.save(sess,model_save_file_path)


def test_model(input_file_path, params, model_save_file_path):
    u1s, u2s, ys = get_input_tensors(input_file_path)

    u1s -= 1
    u2s -= 1

    weights1_1 = tf.Variable(tf.zeros([params['node_num'], params['h_len']]), name='weights1_1')
    weights1_2 = tf.Variable(tf.zeros([params['node_num'], params['h_len']]), name='weights1_2')
    biases1 = tf.Variable(tf.zeros([params['h_len']]), name='biases1')
    h1 = tf.nn.softmax(tf.nn.embedding_lookup(weights1_1, u1s) + tf.nn.embedding_lookup(weights1_2, u2s) + biases1)

    weights2 = tf.Variable(tf.ones([params['h_len'], params['h_len']]), name='weights2')
    biases2 = tf.Variable(tf.zeros([params['h_len']]), name='biases2')
    h2 = tf.nn.softmax(tf.matmul(h1, weights2) + biases2)

    weights3 = tf.Variable(tf.ones([params['h_len'], 1]), name='weight3')
    biases3 = tf.Variable(tf.zeros([1]), name='biases3')
    ys_ = tf.sigmoid(tf.matmul(h2, weights3) + biases3)

    saver = tf.train.Saver({'weights1_1': weights1_1, 'weights1_2': weights1_2, 'biases1': biases1, 'weight2': weights2,
                            'biases2': biases2, 'weights3': weights3, 'biases3': biases3})

    with tf.Session() as sess:
        saver.restore(sess, model_save_file_path)
        # print sess.run(tf.reduce_mean(tf.cast(tf.equal(ys,tf.cast(tf.round(ys_),tf.int32)),"float")))
        ys_ = tf.reduce_sum(ys_, 1)
        print sess.run(h1)
        print sess.run(h2)
        print sess.run(weights1_1)
        print sess.run(weights1_2)
    # print sess.run(tf.reduce_mean(tf.cast(tf.equal(ys,tf.cast(tf.round(ys_),tf.int32)),"float")))


def main():
    train_data_file_path = '../../data/test/small_train_data_csv'  # '../../data/data/train_data_csv'
    test_data_file_path = '../../data/test/tiny_test_data_csv'  # '../../data/tmp/csv'
    params = {'node_num': 10000, 'round': 2000, 'learning_rate': 0.003, 'learning_rate1': 1, 'learning_rate2': 0.001, 'batch_size': 500,
              'h_len': 4}  # {'node_num':1632803,'edge_num':30622564*4,'round':20,'learning_rate':0.01,'batch_size':30000,'h_len':10}
    model_save_file_path = '../../data/test/model_weights.ckpt'
    train_model(train_data_file_path, params, model_save_file_path)
    # test_model(test_data_file_path,params,model_save_file_path)


if __name__ == '__main__':
    # shuffle_batch_test()
    # saver_save_test()
    # saver_restore_test()
    main()
