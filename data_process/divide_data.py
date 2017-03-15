#coding:utf-8
import random

def randomly_divide_data(source_file_path,target_file_path1,target_file_path2,weight1,weight2):
    with open(source_file_path,'r') as source_file,open(target_file_path1,'w') as target_file1,open(target_file_path2,'w') as target_file2:
	list1=[]
	list2=[]
	count1=0
	count2=0
	threshold=weight1/float(weight1+weight2)
	for line in source_file:
	    if random.random()<threshold:
		list1.append(line)
		count1+=1
	    else:
		list2.append(line)
		count2+=1
	print count1
	print count2
	target_file1.writelines(list1)
	target_file2.writelines(list2)

if __name__=='__main__':
    randomly_divide_data('../../data/soc-pokec-relationships.txt','../../data/train_edges','../../data/test_edges',8,2)
