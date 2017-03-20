import tensorflow as tf
import numpy as np

def get_input_tensors(input_file_path):
    list1=[]
    list2=[]
    list3=[]
    with open(input_file_path,'r') as input_file:
	for line in input_file:
	    items=line[0:len(line)-1].split('\t')
	    list1.append(int(items[0]))
	    list2.append(int(items[1]))
	    list3.append(int(items[2]))
    return [tf.convert_to_tensor(list1),tf.convert_to_tensor(list2),tf.convert_to_tensor(list3)]

def shuffle_batch_test():
    filenames=['../../data/test/train_data_csv_test']
    filename_queue=tf.train.string_input_producer(filenames)
    reader=tf.TextLineReader()
    key,value=reader.read(filename_queue)
    u1,u2,label=tf.decode_csv(value,record_defaults=[[0],[0],[0.0]])
    u1_batch,u2_batch,label_batch=tf.train.shuffle_batch([u1,u2,label],batch_size=1,capacity=20,min_after_dequeue=10)
    with tf.Session() as sess:
        coord=tf.train.Coordinator()
        thread=tf.train.start_queue_runners(coord=coord)
        for i in range(40):
            u1val,u2val,lval=sess.run([u1_batch,u2_batch,label_batch])
	    print u1val,u2val,lval
        coord.request_stop()
        coord.join(thread)

def train_model(input_file_path,params):
    filenames=[input_file_path]
    filename_queue=tf.train.string_input_producer(filenames)
    reader=tf.TextLineReader()
    key,value=reader.read(filename_queue)
    u1,u2,y=tf.decode_csv(value,record_defaults=[[0],[0],[0.0]])

    u1s,u2s,ys=tf.train.shuffle_batch([u1,u2,y],batch_size=params['batch_size'],capacity=3000,min_after_dequeue=1000)

    #u1s=tf.placeholder(tf.int32,shape=[None,])
    #u2s=tf.placeholder(tf.int32,shape=[None,])
    #ys=tf.placeholder(tf.float32,shape=[None,])
    #oh_u1s=tf.one_hot(u1s-1,depth=params['node_num'])
    #oh_u2s=tf.one_hot(u2s-1,depth=params['node_num'])

    u1s -= 1
    u2s -= 1

    weights1_1=tf.Variable(tf.ones([params['node_num'],params['h_len']]))
    weights1_2=tf.Variable(tf.ones([params['node_num'],params['h_len']]))
    biases1=tf.Variable(tf.zeros([params['h_len']]))
    h1=tf.nn.softmax(tf.nn.embedding_lookup(weights1_1, u1s)+tf.nn.embedding_lookup(weights1_2, u2s)+biases1)
    
    weights2=tf.Variable(tf.ones([params['h_len'],params['h_len']]))
    biases2=tf.Variable(tf.zeros([params['h_len']]))
    h2=tf.nn.softmax(tf.matmul(h1,weights2)+biases2)

    weights3=tf.Variable(tf.ones([params['h_len'],1]))
    biases3=tf.Variable(tf.zeros([1]))
    ys_=tf.sigmoid(tf.matmul(h2,weights3)+biases3)

    cross_entropy=-tf.reduce_sum(ys*tf.log(ys_))

    train_step=tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(cross_entropy)

    with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
        coord=tf.train.Coordinator()
        thread=tf.train.start_queue_runners(coord=coord)
	for i in range(int(float(params['round'])*params['edge_num']/params['batch_size'])):
	    train_step.run()
	    if i%100==0:
	        print 'step count: %db'%(i/100)
	coord.request_stop()
        coord.join(thread)

def main():
    input_file_path='../../data/data/train_data_csv'#'../../data/data/train_data_csv'
    params={'node_num':1632803,'edge_num':30622564,'round':20,'learning_rate':0.01,'batch_size':60000,'h_len':10}#{'node_num':1632803,'edge_num':30622564,'round':20,'learning_rate':0.01,'batch_size':60000,'h_len':10}
    train_model(input_file_path,params)

if __name__=='__main__':
    #shuffle_batch_test()
    main()
