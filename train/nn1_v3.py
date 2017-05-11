import tensorflow as tf
import numpy as np

def train_model(input_file_path, params, model_save_file_path):
    dtrain = np.loadtxt(input_file_path, delimiter=',', dtype=np.int32)
    dtrain -= [1, 1, 0]

    u1s = tf.placeholder(tf.int32, name='u1s')
    u2s = tf.placeholder(tf.int32, name='u2s')
    ys = tf.placeholder(tf.float32, name='ys')

    embeddings1 = tf.Variable(tf.random_normal([params['node_num'], params['embedding_size']], mean = 0, stddev = 1), name='embeddings1')
    embeddings2 = tf.Variable(tf.random_normal([params['node_num'], params['embedding_size']], mean = 0, stddev = 1), name='embeddings2')
    e_biases = tf.Variable(tf.zeros([params['embedding_size'] * 2]), name='e_biases')
    e1 = tf.nn.embedding_lookup(embeddings1, u1s)
    e2 = tf.nn.embedding_lookup(embeddings2, u2s)
    h0 = tf.concat(1, [e1, e2]) + e_biases

    weights1 = tf.Variable(tf.random_normal([params['embedding_size'] * 2, params['h1_size']], mean = 0, stddev =  1), name='weights1')
    biases1 = tf.Variable(tf.zeros([params['h1_size']]), name='biases1')
    h1 = tf.matmul(h0, weights1) + biases1

    weights2 = tf.Variable(tf.random_normal([params['h1_size'], params['h2_size']], mean = 0, stddev = 1))
    biases2 = tf.Variable(tf.zeros([params['h2_size']]))
    h2 = tf.matmul(h1, weights2) + biases2

    weights3 = tf.Variable(tf.random_normal([params['h2_size'], 1], mean=0, stddev=1))
    biases3 = tf.Variable(tf.zeros([1]))
    ys_ = tf.reduce_sum(tf.matmul(h2, weights3) + biases3, 1)
    # ys_ = tf.reduce_sum(tf.sigmoid(tf.matmul(h2, weights3) + biases3), 1)

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(ys_, ys))
    # cross_entropy = tf.reduce_mean(tf.sqrt(tf.square(ys_ - ys)))

    train_step = tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(cross_entropy)
    # var_list1 = [weights1_1, weights1_2]
    # var_list2 = [weights2_2, biases1, biases2]
    # op1 = tf.train.GradientDescentOptimizer(params['learning_rate1']).minimize(cross_entropy, var_list = var_list1)
    # op2 = tf.train.GradientDescentOptimizer(params['learning_rate2']).minimize(cross_entropy, var_list = var_list2)
    # train_step = tf.group(op1, op2)

    auc, _= tf.contrib.metrics.streaming_auc(tf.sigmoid(ys_), ys)
    # auc, _= tf.contrib.metrics.streaming_auc(ys_, ys)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys,tf.int32),tf.cast(tf.sigmoid(ys_),tf.int32)),tf.float32))
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(ys,tf.int32),tf.cast(ys_,tf.int32)),tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        round_count=0
        np.random.shuffle(dtrain)
        data_size = dtrain.shape[0]

        print 'round%d: %f' % (round_count, sess.run(cross_entropy, feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])}))
        # print sess.run([ys_], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})

        for i in range(int(float(params['round']) * data_size / params['batch_size'])):
            start = i * params['batch_size'] % dtrain.shape[0]
            end = (i + 1) * params['batch_size'] % dtrain.shape[0]
            if end <= start:
                round_count+=1
                np.random.shuffle(dtrain)
                auc_v, accuracy_v, ce, y_pres, y_trues, weights1_v, weights2_v, weights3_v = sess.run([auc, accuracy, cross_entropy, tf.round(tf.sigmoid(ys_)), ys, weights1, weights2, weights3], feed_dict={u1s: dtrain[:, 0], u2s: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
                print 'round%d: %.3f' % (round_count,ce)
                print auc_v, accuracy_v, y_pres.mean()
                # for j in range(y_pres.shape[0]):
                #     print y_pres[j],y_trues[j]
                # print weights1_1_v, weights2_1_v
                continue
            sess.run(train_step, feed_dict={u1s: dtrain[start:end, 0], u2s: dtrain[start:end, 1], ys: np.float32(dtrain[start:end, 2])})

def main():
    train_data_file_path = '../../data/test/small_train_data_v2_csv'
    params = {'node_num': 10000, 'round': 10, 'learning_rate': 0.02, 'batch_size': 500, 'learning_rate1': 1, 'learning_rate2': 0.001,
              'embedding_size': 10, 'h1_size': 8, 'h2_size':4}
    model_save_file_path = '../../data/test/model_weights.ckpt'
    train_model(train_data_file_path, params, model_save_file_path)

if __name__ == '__main__':
    main()
