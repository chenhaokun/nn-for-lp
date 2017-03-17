#coding:utf-8
import random

def randomly_divide_data(source_file_path,target_file_path1,target_file_path2,tt_rate,np_rate,node_num):
    n_edge_dict={}
    n_edge_list=[]
    p_edge_count=0
    n_edge_count=0
    print('step1...')
    with open(source_file_path,'r') as source_file:
	for line in source_file:
	    tmpstr=line[0:len(line)-1]
	    n_edge_dict[tmpstr]=''
	    p_edge_count+=1
    n_edge_num=np_rate*p_edge_count
    print('step2...')
    while 1:
	line='%d\t%d'%(random.randint(1,node_num),random.randint(1,node_num))
	if line not in n_edge_dict:
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
	print count1
	print count2
	target_file1.writelines(list1)
	target_file2.writelines(list2)

def tsv_to_csv(source_file_path,target_file_path):
    with open(source_file_path,'r') as source_file,open(target_file_path,'w') as target_file:
	target_list=[]
	for line in source_file:
	    newline=','.join(line.split('\t'))
	    target_list.append(newline)
	target_file.writelines(target_list)	     

if __name__=='__main__':
    #randomly_divide_data('../../data/raw/soc-pokec-relationships.txt','../../data/data/train_data','../../data/data/test_data',4,4,1632803)
    tsv_to_csv('../../data/data/train_data','../../data/data/train_data_csv')
    tsv_to_csv('../../data/data/test_data','../../data/data/test_data_csv')
