#coding:utf-8

def get_up_bound_id(source_file_path):
    with open(source_file_path,'r') as source_file:
	max=0
	count=0
	for line in source_file:
	    items=(line[0:len(line)-1]).split('\t')
	    i=int(items[0])
	    j=int(items[1])
	    if i>max:
		max=i
	    if j>max:
		max=j
	    count+=1
	print max
	print count

if __name__=='__main__':
    #get_up_bound_id('../../data/data/soc-pokec-relationships.txt')
    get_up_bound_id('../../data/test/tiny_data')
