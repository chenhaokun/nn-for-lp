#coding:utf-8
import pickle
import os

#[public,completion_percentage,gender,age,height,weight] 
def gen_node_based_feature(source_file_path,target_file_path):
    with open(source_file_path,'r') as source_file:
        user_feature_dict={}
	count=0
	for line in source_file:
	    items=(line[0:len(line)-1]).split('\t')
	    #print items
	    user_id=get_int(items[0])
	    if user_id not in user_feature_dict:
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
	    if count%10000==0:
		print 'processed count: %dw'%(count/10000)
	store_dict(user_feature_dict,target_file_path)

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
    source_file_path='../../data/soc-pokec-profiles.txt'
    target_file_path='../../data/user_feature_dict'
    gen_node_based_feature(source_file_path,target_file_path)
    d=load_dict('../../data/user_feature_dict')
    count=0
    for item in d:
	count+=1
	print d[item]
    print count
