#coding:utf-8
import random
import pickle
import os

#negtive edge is randomly sampled
def randomly_divide_data(source_file_path,target_file_path1,target_file_path2,tt_rate,np_rate,node_num):
    p_edge_dict={}
    n_edge_list=[]
    p_edge_count=0
    n_edge_count=0
    print('step1...')
    with open(source_file_path,'r') as source_file:
        for line in source_file:
            tmpstr=line[0:len(line)-1]
            p_edge_dict[tmpstr]=''
            p_edge_count+=1
    n_edge_num=np_rate*p_edge_count
    print('step2...')
    while 1:
        line='%d\t%d'%(random.randint(1,node_num),random.randint(1,node_num))
        if line not in p_edge_dict:
            n_edge_list.append(line+'\t%d\n'%0)
            n_edge_count+=1
            if n_edge_count==n_edge_num:
                break
    print('step3...')
    with open(source_file_path,'r') as source_file,open(target_file_path1,'w') as target_file1,open(target_file_path2,'w') as target_file2:
        list1=[]
        list2=[]
        count1=0
        count2=0
        n_index=0
        threshold=tt_rate/float(tt_rate+1)
        for line in source_file:
            if random.random()<threshold:
                list1.append(line[0:len(line)-1]+'\t%d\n'%1)
                count1+=1
                for i in range(np_rate):
                    list1.append(n_edge_list[n_index])
                    n_index+=1
            else:
                list2.append(line[0:len(line)-1]+'\t%d\n'%1)
                count2+=1
                for i in range(np_rate):
                    list2.append(n_edge_list[n_index])
                    n_index+=1
        print count1#positive train
        print count2#positive test
        target_file1.writelines(list1)
        target_file2.writelines(list2)

#negtive edge is sampled with one node fixed to the positive one
def randomly_divide_data_v2(source_file_path,target_file_path1,target_file_path2,tt_rate,np_rate,node_num):
    p_edge_dict={}
    n_edge_dict={}
    n_edge_list=[]
    p_edge_count=0
    n_edge_count=0
    print('step1...')
    with open(source_file_path, 'r') as source_file:
        for line in source_file:
            tmpstr=line[0:len(line)-1]
            p_edge_dict[tmpstr]=''
            p_edge_count+=1
    n_edge_num=np_rate*p_edge_count
    print('step2...')
    # while 1:
    #     line='%d\t%d'%(random.randint(1,node_num),random.randint(1,node_num))
    #     if line not in p_edge_dict:
    #         n_edge_list.append(line+'\t%d\n'%0)
    #         n_edge_count+=1
    #         if n_edge_count==n_edge_num:
    #             break
    with open(source_file_path, 'r') as source_file:
        for line in source_file:
            items = line[0: len(line) - 1].split('\t')
            u1 = int(items[0])
            u2 = int(items[1])
            for j in range((np_rate + 1) / 2):
                while 1:
                    new_line = '%d\t%d' % (u1, random.randint(1, node_num))
                    if new_line not in p_edge_dict and new_line not in n_edge_dict:
                        n_edge_list.append(new_line + '\t%d\n' % 0)
                        n_edge_dict[new_line] = ''
                        break
                while 1:
                    new_line = '%d\t%d' % (random.randint(1, node_num), u2)
                    if new_line not in p_edge_dict and new_line not in n_edge_dict:
                        n_edge_list.append(new_line + '\t%d\n' % 0)
                        n_edge_dict[new_line] = ''
                        break
    print('step3...')
    with open(source_file_path,'r') as source_file,open(target_file_path1,'w') as target_file1,open(target_file_path2,'w') as target_file2:
        list1=[]
        list2=[]
        count1=0
        count2=0
        n_index=0
        threshold=tt_rate/float(tt_rate+1)
        for line in source_file:
            if random.random()<threshold:
                list1.append(line[0:len(line)-1]+'\t%d\n'%1)
                count1+=1
                for i in range(np_rate):
                    list1.append(n_edge_list[n_index])
                    n_index+=1
            else:
                list2.append(line[0:len(line)-1]+'\t%d\n'%1)
                count2+=1
                for i in range(np_rate):
                    list2.append(n_edge_list[n_index])
                    n_index+=1
        print count1#positive train
        print count2#positive test
        target_file1.writelines(list1)
        target_file2.writelines(list2)

#negtive edge is randomly sampled, when train data negtive edge may contain positive edge in test data
def randomly_divide_data_v3(source_file_path,target_file_path1,target_file_path2,tt_rate,np_rate,node_num):
    p_train = {}
    p_test = {}
    n_train = {}
    n_test = {}
    p_train_count = 0
    p_test_count = 0
    n_train_count = 0
    n_test_count = 0
    print 'generating positive data...'
    with open(source_file_path, 'r') as source_file:
        thredhold = float(tt_rate)/(tt_rate + 1)
        for line in source_file:
            if random.random() < thredhold:
                p_train[line] = ''
                p_train_count += 1
            else:
                p_test[line] = ''
                p_test_count += 1
    n_train_target = p_train_count * np_rate
    n_test_target = p_test_count * np_rate
    print 'generating negative data...'
    while(n_train_count<n_train_target):
        line = '%d\t%d\n' % (random.randint(1, node_num), random.randint(1, node_num))
        if line not in p_train:
            n_train[line] = ''
            n_train_count += 1
    while(n_test_count<n_test_target):
        line = '%d\t%d\n' % (random.randint(1, node_num), random.randint(1, node_num))
        if line not in p_train and line not in p_test and line not in n_train:
            n_test[line] = ''
            n_test_count += 1
    print 'writing data...'
    with open(target_file_path1, 'w') as target_file1:
        result_list = []
        for line in p_train:
            result_list.append(line[0:-1] + '\t1\n')
        for line in n_train:
            result_list.append(line[0:-1] + '\t0\n')
        target_file1.writelines(result_list)
    with open(target_file_path2, 'w') as target_file2:
        result_list = []
        for line in p_test:
            result_list.append(line[0:-1] + '\t1\n')
        for line in n_test:
            result_list.append(line[0:-1] + '\t0\n')
        target_file2.writelines(result_list)
    print 'p_train_count:\t%d' % p_train_count
    print 'p_test_count:\t%d' % p_test_count
    print 'n_train_count:\t%d' % n_train_count
    print 'n_test_count:\t%d' % n_test_count

#negtive edge is randomly sampled, when train data negtive edge may contain positive edge in test data, store hop2 train positive data
def randomly_divide_data_v4(source_file_path,target_file_path1,target_file_path2,target_file_path3,tt_rate,np_rate,node_num):
    p_train = {}
    p_test = {}
    n_train = {}
    n_test = {}
    p_train_count = 0
    p_test_count = 0
    n_train_count = 0
    n_test_count = 0
    print 'generating positive data...'
    with open(source_file_path, 'r') as source_file:
        thredhold = float(tt_rate)/(tt_rate + 1)
        for line in source_file:
            if random.random() < thredhold:
                p_train[line] = ''
                p_train_count += 1
            else:
                p_test[line] = ''
                p_test_count += 1
    n_train_target = p_train_count * np_rate
    n_test_target = p_test_count * np_rate
    print 'generating negative data...'
    while(n_train_count<n_train_target):
        a = random.randint(1, node_num)
        b = random.randint(1, node_num)
        line = '%d\t%d\n' % (a, b)
        if a!=b and line not in p_train and line not in n_train:
            n_train[line] = ''
            n_train_count += 1
    while(n_test_count<n_test_target):
        a = random.randint(1, node_num)
        b = random.randint(1, node_num)
        line = '%d\t%d\n' % (a, b)
        if a!=b and line not in p_train and line not in p_test and line not in n_train and line not in n_test:
            n_test[line] = ''
            n_test_count += 1
    print 'writing data...'
    with open(target_file_path1, 'w') as target_file1:
        result_list = []
        for line in p_train:
            result_list.append(line[0:-1] + '\t1\n')
        for line in n_train:
            result_list.append(line[0:-1] + '\t0\n')
        target_file1.writelines(result_list)
    with open(target_file_path2, 'w') as target_file2:
        result_list = []
        for line in p_test:
            result_list.append(line[0:-1] + '\t1\n')
        for line in n_test:
            result_list.append(line[0:-1] + '\t0\n')
        target_file2.writelines(result_list)
    with open(target_file_path3, 'w') as target_file3:
        result_list = []
        for line in p_train:
            result_list.append(line)
        target_file3.writelines(result_list)
    print 'p_train_count:\t%d' % p_train_count
    print 'p_test_count:\t%d' % p_test_count
    print 'n_train_count:\t%d' % n_train_count
    print 'n_test_count:\t%d' % n_test_count

def add_negative_samples_and_label(source_file_path, target_file_path, np_rate, node_num):
    p_dict = {}
    n_dict = {}
    p_count = 0
    n_count = 0
    n_target = 0
    with open(source_file_path, 'r') as source_file:
        for line in source_file:
            p_dict[line] = ''
            p_count += 1
    n_target = p_count * np_rate
    while(n_count<n_target):
        a = random.randint(1, node_num)
        b = random.randint(1, node_num)
        line = '%d\t%d\n' % (a, b)
        if a!=b and line not in p_dict and line not in n_dict:
            n_dict[line] = ''
            n_count += 1
            if n_count%1000000 == 0:
                print n_count / 1000000
    result_list = []
    for line in p_dict:
        result_list.append(line[0:-1] + '\t1\n')
    for line in n_dict:
        result_list.append(line[0:-1] + '\t0\n')
    with open(target_file_path, 'w') as target_file:
        target_file.writelines(result_list)

def add_one_to_node_id(source_file_path, target_file_path):
    with open(source_file_path, 'r') as source_file, open(target_file_path, 'w') as target_file:
        result_list = []
        for line in source_file:
            items = line[0:-1].split('\t')
            newline = '%d\t%d\n' % (int(items[0])+1, int(items[1])+1)
            result_list.append(newline)
        target_file.writelines(result_list)

def to_tsv(source_file_path, target_file_path):
    with open(source_file_path, 'r') as source_file, open(target_file_path, 'w') as target_file:
        result_list = []
        for line in source_file:
            items = line.split(' ')
            newline = items[0] + '\t' + items[1] + '\n'
            result_list.append(newline)
        target_file.writelines(result_list)

def tsv_to_csv(source_file_path,target_file_path):
    with open(source_file_path,'r') as source_file,open(target_file_path,'w') as target_file:
        target_list=[]
        for line in source_file:
            newline=','.join(line.split('\t'))
            target_list.append(newline)
        target_file.writelines(target_list)

def get_small_data(source_file_path,target_file_path,max_node_num):
    with open(source_file_path,'r') as source_file,open(target_file_path,'w') as target_file:
        t_list=[]
        for line in source_file:
            items=line[0:len(line)-1].split('\t')
            if(int(items[0])<=max_node_num and int(items[1])<=max_node_num):
                t_list.append(line)
        target_file.writelines(t_list)

def extend_graph(source_file_path, target_file_path):
    friends_dict = {}
    result_lists = []
    print 'step1...'
    with open(source_file_path, 'r') as source_file:
        for line in source_file:
            items=line[0:len(line)-1].split('\t')
            if items[0] not in friends_dict:
                friends_dict[items[0]] = []
                friends_dict[items[0]].append(items[1])
            else:
                friends_dict[items[0]].append(items[1])
    print 'step2...'
    with open(source_file_path, 'r') as source_file, open(target_file_path, 'w') as target_file:
        for line in source_file:
            items=line[0:len(line)-1].split('\t')
            if items[1] not in friends_dict:
                continue
            for tmp in friends_dict[items[1]]:
                result_lists.append(items[0]+'\t'+tmp+'\n')
        target_file.writelines(result_lists)

#delete repeated edge, ring edge
def extend_graph_v2(source_file_path, target_file_path):
    friends_dict = {}
    result_dict = {}
    result_list = []
    print 'step1...'
    with open(source_file_path, 'r') as source_file:
        for line in source_file:
            items=line[0:len(line)-1].split('\t')
            if items[0] not in friends_dict:
                friends_dict[items[0]] = []
                friends_dict[items[0]].append(items[1])
            else:
                friends_dict[items[0]].append(items[1])
    print 'step2...'
    with open(source_file_path, 'r') as source_file, open(target_file_path, 'w') as target_file:
        for line in source_file:
            items=line[0:len(line)-1].split('\t')
            if items[1] not in friends_dict:
                continue
            for tmp in friends_dict[items[1]]:
                if tmp != items[0]:
                    result_dict[items[0]+'\t'+tmp+'\n'] = ''
        for line in result_dict:
            result_list.append(line)
        print len(result_list)
        target_file.writelines(result_list)

def gen_libsvm_data(source_file_path, target_file_path, uf_list_file_path):
    with open(source_file_path,'r') as read_file, open(target_file_path,'w') as write_file:
        result_list = []
        feature_list = load_dict(uf_list_file_path)
        for line in read_file:
            items=line[0:len(line)-1].split(',')
            u1 = int(items[0]) - 1
            u2 = int(items[1]) - 1
            newline = '%d 1:%d 2:%d'%(int(items[2]), u1, u2)
            count = 3
            for i in range(6):
                newline += ' %d:%.2f' % (count, float(feature_list[u1][i]))
                count += 1
            for i in range(6):
                newline += ' %d:%.2f' % (count, float(feature_list[u2][i]))
                count += 1
            newline += '\n'
            result_list.append(newline)
        write_file.writelines(result_list)

def load_dict(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path,'rb') as f:
        return pickle.load(f)

def gen_full_tt_data(source_file_path, target_file_path, uf_file_path):#for exp17
    with open(source_file_path, 'r') as source_file, open(target_file_path, 'w') as write_file:
        uf_list = load_dict(uf_file_path)
        result_list = []
        for line in source_file:
            items = line[0:len(line)-1].split(',')
            u1 = int(items[0]) - 1
            u2 = int(items[1]) - 1
            label = int(items[2])
            l1 = uf_list[u1]
            l2 = uf_list[u2]
            new_line = '%d,%d,' % (u1, u2)
            for i in range(len(l1)):
                new_line += '%d,%d,' % (l1[i], l2[i])
            new_line += '%d\n' % label
            result_list.append(new_line)
        write_file.writelines(result_list)

if __name__=='__main__':
    # randomly_divide_data('../../data/raw/soc-pokec-relationships.txt','../../data/data/train_data','../../data/data/test_data',4,4,1632803)
    # tsv_to_csv('../../data/data/train_data','../../data/data/train_data_csv')
    # tsv_to_csv('../../data/data/test_data','../../data/data/test_data_csv')
    # get_small_data('../../data/raw/soc-pokec-relationships.txt','../../data/test/small_data',10000)
    # randomly_divide_data('../../data/test/small_data','../../data/test/small_train_data','../../data/test/small_test_data',4,4,10000)
    # tsv_to_csv('../../data/test/small_train_data','../../data/test/small_train_data_csv')
    # tsv_to_csv('../../data/test/small_test_data','../../data/test/small_test_data_csv')
    # get_small_data('../../data/test/small_data','../../data/test/tiny_data',100)
    # randomly_divide_data('../../data/test/small_data','../../data/test/small_train_data_v2','../../data/test/small_test_data_v2',4,1,10000)
    # tsv_to_csv('../../data/test/small_train_data_v2','../../data/test/small_train_data_v2_csv')
    # tsv_to_csv('../../data/test/small_test_data_v2','../../data/test/small_test_data_v2_csv')
    # extend_graph('../../data/test/small_data', '../../data/test/small_data_distance2')
    # gen_libsvm_data('../../data/test/small_train_data_csv', '../../data/test/small_train_data_libsvm', '../../data/test/small_data_user_feature_list')
    # gen_libsvm_data('../../data/test/small_test_data_csv', '../../data/test/small_test_data_libsvm', '../../data/test/small_data_user_feature_list')
    # randomly_divide_data_v2('../../data/test/small_data', '../../data/test/small_train_data_v3', '../../data/test/small_test_data_v3', 4, 4, 10000)
    # tsv_to_csv('../../data/test/small_train_data_v3','../../data/test/small_train_data_v3_csv')
    # tsv_to_csv('../../data/test/small_test_data_v3','../../data/test/small_test_data_v3_csv')
    # gen_full_tt_data('../../data/test/small_train_data_v3_csv', '../../data/test/small_train_data_v4_csv', '../../data/test/small_data_uf_bucket_list')
    # gen_full_tt_data('../../data/test/small_test_data_v3_csv', '../../data/test/small_test_data_v4_csv', '../../data/test/small_data_uf_bucket_list')
    # get_small_data('../../data/raw/soc-pokec-relationships.txt', '../../data/test/m_data', 100000)
    # randomly_divide_data_v3('../../data/test/m_data', '../../data/test/m_train_data_v1', '../../data/test/m_test_data_v1', 9, 4, 100000)
    # tsv_to_csv('../../data/test/m_train_data_v1','../../data/test/m_train_data_v1_csv')
    # tsv_to_csv('../../data/test/m_test_data_v1','../../data/test/m_test_data_v1_csv')
    # gen_full_tt_data('../../data/test/m_train_data_v1_csv', '../../data/test/m_train_data_v2_csv', '../../data/test/m_data_uf_bucket_list')
    # gen_full_tt_data('../../data/test/m_test_data_v1_csv', '../../data/test/m_test_data_v2_csv', '../../data/test/m_data_uf_bucket_list')
    # extend_graph_v2('../../data/test/small_data', '../../data/test/small_data_distance2_v2')
    # randomly_divide_data_v3('../../data/test/small_data_distance2_v2', '../../data/test/small_hop2_train_data_v1', '../../data/test/small_hop2_test_data_v1', 9, 4, 10000)
    # p_train_count:    2270731
    # p_test_count:    252801
    # n_train_count:    9082924
    # n_test_count:    1011204
    # tsv_to_csv('../../data/test/small_hop2_train_data_v1', '../../data/test/small_hop2_train_data_v1_csv')
    # tsv_to_csv('../../data/test/small_hop2_test_data_v1','../../data/test/small_hop2_test_data_v1_csv')
    # extend_graph_v2('../../data/test/m_data', '../../data/test/m_data_distance2_v2')
    # randomly_divide_data_v3('../../data/test/m_data_distance2_v2', '../../data/test/m_hop2_train_data_v1', '../../data/test/m_hop2_test_data_v1', 9, 4, 100000)
    # tsv_to_csv('../../data/test/m_hop2_train_data_v1', '../../data/test/m_hop2_train_data_v1_csv')
    # tsv_to_csv('../../data/test/m_hop2_test_data_v1', '../../data/test/m_hop2_test_data_v1_csv')

    #for nn_v5
    # randomly_divide_data_v4('../../data/test/small_data', '../../data/test/small_train_data_v5', '../../data/test/small_test_data_v5', '../../data/test/small_train_positive_data_v5', 4, 4, 10000)
    # extend_graph_v2('../../data/test/small_train_positive_data_v5', '../../data/tmp/small_hop2_train_positive_data_v5')
    # add_negative_samples_and_label('../../data/tmp/small_hop2_train_positive_data_v5', '../../data/test/small_hop2_train_data_v2', 4, 10000)
    # tsv_to_csv('../../data/test/small_train_data_v5', '../../data/test/small_train_data_v5_csv')
    # tsv_to_csv('../../data/test/small_test_data_v5', '../../data/test/small_test_data_v5_csv')
    # tsv_to_csv('../../data/test/small_hop2_train_data_v2', '../../data/test/small_hop2_train_data_v2_csv')

    # randomly_divide_data_v4('../../data/test/m_data', '../../data/test/m_train_data_v5', '../../data/test/m_test_data_v5', '../../data/test/m_train_positive_data_v5', 9, 4, 100000)
    # extend_graph_v2('../../data/test/m_train_positive_data_v5', '../../data/tmp/m_hop2_train_positive_data_v5')
    # add_negative_samples_and_label('../../data/tmp/m_hop2_train_positive_data_v5', '../../data/test/m_hop2_train_data_v2', 4, 100000)
    # tsv_to_csv('../../data/test/m_train_data_v5', '../../data/test/m_train_data_v5_csv')
    # tsv_to_csv('../../data/test/m_test_data_v5', '../../data/test/m_test_data_v5_csv')
    # tsv_to_csv('../../data/test/m_hop2_train_data_v2', '../../data/test/m_hop2_train_data_v2_csv')

    # get_small_data('../../data/test/m_data', '../../data/test/hm_data', 50000)
    # randomly_divide_data_v4('../../data/test/hm_data', '../../data/test/hm_train_data_v5', '../../data/test/hm_test_data_v5', '../../data/test/hm_train_positive_data_v5', 9, 4, 50000)
    # extend_graph_v2('../../data/test/hm_train_positive_data_v5', '../../data/tmp/hm_hop2_train_positive_data_v5')
    # add_negative_samples_and_label('../../data/tmp/hm_hop2_train_positive_data_v5', '../../data/test/hm_hop2_train_data_v2', 4, 50000)
    # tsv_to_csv('../../data/test/hm_train_data_v5', '../../data/test/hm_train_data_v5_csv')
    # tsv_to_csv('../../data/test/hm_test_data_v5', '../../data/test/hm_test_data_v5_csv')
    # tsv_to_csv('../../data/test/hm_hop2_train_data_v2', '../../data/test/hm_hop2_train_data_v2_csv')

    # add_one_to_node_id('../../data/raw/Email-EuAll.txt', '../../data/test/eu_data')
    # randomly_divide_data_v4('../../data/test/eu_data', '../../data/test/eu_train_data_v1', '../../data/test/eu_test_data_v1', '../../data/test/eu_train_positive_data_v1', 9, 4, 265214)
    # extend_graph_v2('../../data/test/eu_train_positive_data_v1', '../../data/tmp/eu_hop2_train_positive_data_v1')
    # add_negative_samples_and_label('../../data/tmp/eu_hop2_train_positive_data_v1', '../../data/test/eu_hop2_train_data_v1', 4, 265214)
    # tsv_to_csv('../../data/test/eu_train_data_v1', '../../data/test/eu_train_data_v1_csv')
    # tsv_to_csv('../../data/test/eu_test_data_v1', '../../data/test/eu_test_data_v1_csv')
    # tsv_to_csv('../../data/test/eu_hop2_train_data_v1', '../../data/test/eu_hop2_train_data_v1_csv')

    # add_one_to_node_id('../../data/raw/Email-Enron.txt', '../../data/test/enron_data')
    # randomly_divide_data_v4('../../data/test/enron_data', '../../data/test/enron_train_data_v1', '../../data/test/enron_test_data_v1', '../../data/test/enron_train_positive_data_v1', 9, 4, 36692)
    # extend_graph_v2('../../data/test/enron_train_positive_data_v1', '../../data/tmp/enron_hop2_train_positive_data_v1')
    # add_negative_samples_and_label('../../data/tmp/enron_hop2_train_positive_data_v1', '../../data/test/enron_hop2_train_data_v1', 4, 36692)
    # tsv_to_csv('../../data/test/enron_train_data_v1', '../../data/test/enron_train_data_v1_csv')
    # tsv_to_csv('../../data/test/enron_test_data_v1', '../../data/test/enron_test_data_v1_csv')
    # tsv_to_csv('../../data/test/enron_hop2_train_data_v1', '../../data/test/enron_hop2_train_data_v1_csv')

    # to_tsv('../../data/raw/out.subelj_cora_cora', '../../data/test/cora_data')
    # randomly_divide_data_v4('../../data/test/cora_data', '../../data/test/cora_train_data_v1', '../../data/test/cora_test_data_v1', '../../data/test/cora_train_positive_data_v1', 9, 4, 23166)
    # extend_graph_v2('../../data/test/cora_train_positive_data_v1', '../../data/tmp/cora_hop2_train_positive_data_v1')
    # add_negative_samples_and_label('../../data/tmp/cora_hop2_train_positive_data_v1', '../../data/test/cora_hop2_train_data_v1', 4, 23166)
    # tsv_to_csv('../../data/test/cora_train_data_v1', '../../data/test/cora_train_data_v1_csv')
    # tsv_to_csv('../../data/test/cora_test_data_v1', '../../data/test/cora_test_data_v1_csv')
    # tsv_to_csv('../../data/test/cora_hop2_train_data_v1', '../../data/test/cora_hop2_train_data_v1_csv')

    # add_one_to_node_id('../../data/raw/soc-Epinions1.txt', '../../data/test/epinions_data')
    # randomly_divide_data_v4('../../data/test/epinions_data', '../../data/test/epinions_train_data_v1', '../../data/test/epinions_test_data_v1', '../../data/test/epinions_train_positive_data_v1', 9, 4, 75879)
    # extend_graph_v2('../../data/test/epinions_train_positive_data_v1', '../../data/tmp/epinions_hop2_train_positive_data_v1')
    # add_negative_samples_and_label('../../data/tmp/epinions_hop2_train_positive_data_v1', '../../data/test/epinions_hop2_train_data_v1', 4, 75879)
    # tsv_to_csv('../../data/test/epinions_train_data_v1', '../../data/test/epinions_train_data_v1_csv')
    # tsv_to_csv('../../data/test/epinions_test_data_v1', '../../data/test/epinions_test_data_v1_csv')
    # tsv_to_csv('../../data/test/epinions_hop2_train_data_v1', '../../data/test/epinions_hop2_train_data_v1_csv')

    # get_small_data('../../data/test/small_data', '../../data/tmp/tmp_data', 2000)
    # randomly_divide_data_v4('../../data/tmp/tmp_data', '../../data/tmp/tmp_train_data_v1', '../../data/tmp/tmp_test_data_v1', '../../data/tmp/tmp_train_positive_data_v1', 9, 30, 2000)
    # extend_graph_v2('../../data/tmp/tmp_train_positive_data_v1', '../../data/tmp/tmp_hop2_train_positive_data_v1')
    # add_negative_samples_and_label('../../data/tmp/tmp_hop2_train_positive_data_v1', '../../data/tmp/tmp_hop2_train_data_v1', 4, 2000)
    # tsv_to_csv('../../data/tmp/tmp_train_data_v1', '../../data/tmp/tmp_train_data_v1_csv')
    # tsv_to_csv('../../data/tmp/tmp_test_data_v1', '../../data/tmp/tmp_test_data_v1_csv')
    # tsv_to_csv('../../data/tmp/tmp_hop2_train_data_v1', '../../data/tmp/tmp_hop2_train_data_v1_csv')

    # add_one_to_node_id('../../data/raw/Email-EuAll.txt', '../../data/test/eu_data')
    # randomly_divide_data_v4('../../data/test/eu_data', '../../data/test/eu_train_data_v2', '../../data/test/eu_test_data_v2', '../../data/test/eu_train_positive_data_v2', 9, 30, 265214)
    # extend_graph_v2('../../data/test/eu_train_positive_data_v2', '../../data/tmp/eu_hop2_train_positive_data_v2')
    # add_negative_samples_and_label('../../data/tmp/eu_hop2_train_positive_data_v2', '../../data/test/eu_hop2_train_data_v2', 4, 265214)
    # tsv_to_csv('../../data/test/eu_train_data_v2', '../../data/test/eu_train_data_v2_csv')
    # tsv_to_csv('../../data/test/eu_test_data_v2', '../../data/test/eu_test_data_v2_csv')
    # tsv_to_csv('../../data/test/eu_hop2_train_data_v2', '../../data/test/eu_hop2_train_data_v2_csv')

    # randomly_divide_data_v4('../../data/test/cora_data', '../../data/test/cora_train_data_v2', '../../data/test/cora_test_data_v2', '../../data/test/cora_train_positive_data_v2', 9, 30, 23166)
    # extend_graph_v2('../../data/test/cora_train_positive_data_v2', '../../data/tmp/cora_hop2_train_positive_data_v2')
    # add_negative_samples_and_label('../../data/tmp/cora_hop2_train_positive_data_v2', '../../data/test/cora_hop2_train_data_v2', 4, 23166)
    # tsv_to_csv('../../data/test/cora_train_data_v2', '../../data/test/cora_train_data_v2_csv')
    # tsv_to_csv('../../data/test/cora_test_data_v2', '../../data/test/cora_test_data_v2_csv')
    # tsv_to_csv('../../data/test/cora_hop2_train_data_v2', '../../data/test/cora_hop2_train_data_v2_csv')

    # randomly_divide_data_v4('../../data/test/enron_data', '../../data/test/enron_train_data_v2', '../../data/test/enron_test_data_v2', '../../data/test/enron_train_positive_data_v2', 9, 30, 36692)
    # extend_graph_v2('../../data/test/enron_train_positive_data_v2', '../../data/tmp/enron_hop2_train_positive_data_v2')
    # add_negative_samples_and_label('../../data/tmp/enron_hop2_train_positive_data_v2', '../../data/test/enron_hop2_train_data_v2', 4, 36692)
    # tsv_to_csv('../../data/test/enron_train_data_v2', '../../data/test/enron_train_data_v2_csv')
    # tsv_to_csv('../../data/test/enron_test_data_v2', '../../data/test/enron_test_data_v2_csv')
    # tsv_to_csv('../../data/test/enron_hop2_train_data_v2', '../../data/test/enron_hop2_train_data_v2_csv')
