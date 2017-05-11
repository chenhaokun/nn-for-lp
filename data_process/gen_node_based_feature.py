#coding:utf-8
import pickle
import os
import numpy as np
from matplotlib import pyplot as plt

#[public,completion_percentage,gender,age,height,weight] 
def gen_node_based_feature(source_file_path, target_file_path, max_node_num):
    with open(source_file_path,'r') as source_file:
        user_feature_dict={}
        count=0
        for line in source_file:
            items=(line[0:len(line)-1]).split('\t')
            #print items
            user_id=get_int(items[0])
            if user_id <= max_node_num and user_id not in user_feature_dict:
                features=[]
                features.append(get_int(items[1]))
                features.append(get_int(items[2]))
                features.append(get_int(items[3]))
                features.append(get_int(items[7]))
                tmp=items[8].split(' ')
                if len(tmp)>=3:
                    features.append(get_int(tmp[0]))
                    features.append(get_int(tmp[2]))
                else:
                    features.append(0)
                    features.append(0)
                user_feature_dict[user_id]=features
            count+=1
            if count%1000==0:
                print 'processed count: %dk'%(count/1000)
	    store_dict(user_feature_dict,target_file_path)

def gen_sub_node_based_feature(source_file_path, target_file_path, max_node_num):
    uf_dict = load_dict(source_file_path)
    feature_list = []
    for i in range(max_node_num):
        feature_list.append(uf_dict[i+1])
    store_dict(feature_list, target_file_path) # in fact, it is store list

def gen_bucket_feature(feature_list_file_path, target_file_path):
    new_list = []
    feature_list = load_dict(feature_list_file_path)
    for tlist in feature_list:
        items = np.int32(np.asarray(tlist))
        new_items = [items[0]%2, items[1]%100/2, items[2]%2, items[3]%60/2, items[4]%200/2, items[5]%200/2]
        new_list.append(new_items)
    store_dict(new_list, target_file_path)

def show_histograms(bins, data):
    print max(data)
    plt.xlim([min(data), max(data)+1])
    plt.hist(data, bins = bins)
    plt.show()

def get_int(s):
    try:
	return int(s)
    except Exception,e:
	return 0

def store_dict(d,file_path):
    with open(file_path,'wb') as f:
	pickle.dump(d,f,pickle.HIGHEST_PROTOCOL)

def load_dict(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path,'rb') as f:
        return pickle.load(f)

if __name__=='__main__':
    # source_file_path='../../data/raw/soc-pokec-profiles.txt'
    # target_file_path='../../data/data/user_feature_dict'
    # gen_node_based_feature(source_file_path,target_file_path)
    # source_file_path='../../data/raw/soc-pokec-profiles.txt'
    # target_file_path='../../data/test/small_data_user_feature_dict'
    # gen_node_based_feature(source_file_path, target_file_path, 10000)
    # source_file_path = '../../data/data/user_feature_dict'
    # target_file_path='../../data/test/small_data_user_feature_list'
    # gen_sub_node_based_feature(source_file_path, target_file_path, 10000)
    # d=load_dict('../../data/test/small_data_uf_bucket_list')
    # count=0
    # for item in d:
    #     count+=1
    #     print item
    # print count
    # show_histograms([5,10,15,20],[12,20,4,3,6,7,8,9])
    # gen_bucket_feature('../../data/test/small_data_user_feature_list', '../../data/test/small_data_uf_bucket_list')
    # uf = load_dict('../../data/test/small_data_uf_bucket_list')
    # tmp = np.asarray(uf)
    # show_histograms(np.arange(0,200,1), tmp[:, 5])
    source_file_path = '../../data/data/user_feature_dict'
    target_file_path='../../data/test/m_data_user_feature_list'
    gen_sub_node_based_feature(source_file_path, target_file_path, 100000)
    gen_bucket_feature('../../data/test/m_data_user_feature_list', '../../data/test/m_data_uf_bucket_list')
