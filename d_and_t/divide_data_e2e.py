#coding:utf-8
from scipy.sparse.linalg import spsolve
from scipy import sparse
import numpy as np
import random
import pickle
import math
import os

def get_max_node_num(source_file_path):
    print('starting get_max_node_num')

    with open(source_file_path, 'r') as source_file:
        max_num = 0
        count = 0
        for line in source_file:
            items=(line[0:-1]).split('\t')
            i=int(items[0])
            j=int(items[1])
            max_num = max(max_num, i, j)
            count += 1
        print('\tmax_node_num: %d, all positive edge count: %d' % (max_num, count))
        print('get_max_node_num completed\n')
        return max_num

def randomly_divide_data(source_file_path, target_file_path_list, weight_list):
    print('starting randomly_divide_data')

    with open(source_file_path, 'r') as source_file:
        source_list = source_file.readlines()
        random.shuffle(source_list)
        line_num = len(source_list)

        sum = 0
        for weight in weight_list:
            sum += weight

        thredhold_list = [0]
        slide_sum = 0
        for weight in weight_list:
            slide_sum += weight
            thredhold_list.append(int(float(slide_sum) / sum * line_num))

        for i in range(len(target_file_path_list)):
            with open(target_file_path_list[i], 'w') as target_file:
                target_file.writelines(source_list[thredhold_list[i]:thredhold_list[i+1]])
                print('\tfile%d positive edge count: %d'%(i, thredhold_list[i+1]-thredhold_list[i]))

    print('randomly_divide_data completed\n')

def randomly_divide_data_with_accumulation(source_file_path, target_file_path_list, weight_list):
    print('starting randomly_divide_data_with_accumulation')

    with open(source_file_path, 'r') as source_file:
        source_list = source_file.readlines()
        random.shuffle(source_list)
        line_num = len(source_list)

        sum = 0
        for weight in weight_list:
            sum += weight

        thredhold_list = [0]
        slide_sum = 0
        for weight in weight_list:
            slide_sum += weight
            thredhold_list.append(int(float(slide_sum) / sum * line_num))

        for i in range(len(target_file_path_list)):
            with open(target_file_path_list[i], 'w') as target_file:
                if i == (len(weight_list)-1):
                    target_file.writelines(source_list[thredhold_list[i]:thredhold_list[i + 1]])
                else:
                    target_file.writelines(source_list[thredhold_list[0]:thredhold_list[i+1]])
                print('\tfile%d positive edge count: %d'%(i, thredhold_list[i+1]-thredhold_list[i]))

    print('randomly_divide_data_with_accumulation completed\n')

def sample_negative_data(source_file_path, target_file_path, np_rate, node_num, except_list):
    print('starting sample_negative_data')

    except_set = set([])
    n_count = 0
    n_target = 0
    with open(source_file_path, 'r') as source_file:
        source_list = source_file.readlines()
        except_set |= set(source_list)
        n_target = len(source_list) * np_rate

    for path in except_list:
        with open(path, 'r') as except_file:
            except_set |= set(except_file.readlines())

    with open(target_file_path, 'w') as target_file:
        result_list = []
        while n_count<n_target:
            a = random.randint(1, node_num)
            b = random.randint(1, node_num)
            line = '%d\t%d\n' % (a, b)
            if a != b and line not in except_set:
                result_list.append(line)
                except_set.add(line)
                n_count += 1
        target_file.writelines(result_list)

    print('\t negative edge count: %d'%n_count)
    print('sample_negative_data completed\n')

def get_neighbor_set(source_file_path, target_file_path, node_num):
    print('starting get_neighbor_set')

    with open(source_file_path, 'r') as source_file:
        result_list = []
        for i in range(node_num):
            result_list.append([set(), set()])#[in, out]
        for line in source_file:
            items = line[0:-1].split('\t')
            #0 based
            i = int(items[0])-1
            j = int(items[1])-1
            result_list[i][1].add(j)
            result_list[j][0].add(i)
        store_obj(result_list, target_file_path)

    print('get_neighbor_set completed\n')

def get_katz_matrix(source_file_path, target_file_path, node_num, beta, exact_katz):
    print('starting get_katz_matrix')
    A = sparse.lil_matrix((node_num, node_num))
    with open(source_file_path, 'r') as source_file:
        for line in source_file:
            items = line[0:-1].split('\t')
            # 0 based
            i = int(items[0]) - 1
            j = int(items[1]) - 1
            A[i, j] = 1

    if exact_katz:
        I = sparse.identity(node_num).tocsc()
        X = spsolve((I - beta * A), I)
        R = X - I
        # store_obj(R.todok(), target_file_path)
        store_obj(R, target_file_path)
    else:
        bA = beta * A.tocsr()
        R = bA
        bAi = bA.dot(bA)
        for i in range(2):
            R += bAi
            bAi = bAi.dot(bA)
        # store_obj(R.todok(), target_file_path)
        store_obj(R, target_file_path)

    print('get_katz_matrix completed\n')

def get_rwr_matrix(source_file_path, neighbor_set_list_file_path, target_file_path, node_num, restarted_rate, exact_rwr):
    print('starting get_rwr_matrix')
    neighbor_set_list = load_obj(neighbor_set_list_file_path)
    W = sparse.lil_matrix((node_num, node_num))
    with open(source_file_path, 'r') as source_file:
        for line in source_file:
            items = line[0:-1].split('\t')
            # 0 based
            i = int(items[0]) - 1
            j = int(items[1]) - 1
            W[i, j] = 1.0/len(neighbor_set_list[i][1])

    if exact_rwr:
        I = sparse.identity(node_num).tocsc()
        X = spsolve((I - (1-restarted_rate) * W), I)
        R = restarted_rate * X
        store_obj(R, target_file_path)
    else:
        print('wait for coding')

    print('get_rwr_matrix completed\n')

def get_hop2_link(source_file_path, target_file_path, neighbor_set_list_file_path, random_p, h_sample_rate):
    print('starting get_hop2_link')

    neighbor_set_list = load_obj(neighbor_set_list_file_path)
    if neighbor_set_list == -1:
        print 'no neighbor set list file exits'
        return

    # friends_dict = {}
    result_set = set()

    # with open(source_file_path, 'r') as source_file:
    #     for line in source_file:
    #         items=line[0:-1].split('\t')
    #         if items[0] not in friends_dict:
    #             friends_dict[items[0]] = []
    #             friends_dict[items[0]].append(items[1])
    #         else:
    #             friends_dict[items[0]].append(items[1])

    with open(source_file_path, 'r') as source_file, open(target_file_path, 'w') as target_file:
        if random_p:
            for line in source_file:
                items = line[0:-1].split('\t')
                i = int(items[0])
                j = int(items[1])
                if len(neighbor_set_list[j-1][1]) == 0:
                    continue
                for tmp in neighbor_set_list[j-1][1]:
                    if tmp != i:
                        result_set.add('%d\t%d\n' % (i, tmp))
            result_list = [line for line in result_set]
            target_file.writelines(result_list)
        else:
            for line in source_file:
                items = line[0:-1].split('\t')
                i = int(items[0])
                j = int(items[1])
                odegree = len(neighbor_set_list[j-1][1])
                idegree = len(neighbor_set_list[i-1][0])
                if odegree == 0:
                    for l in range(h_sample_rate):
                        result_set.add('%d\t%d\n' % (i, j))
                else:
                    for l in range(h_sample_rate):
                        b = random.sample(neighbor_set_list[j-1][1], 1)[0] + 1
                        result_set.add('%d\t%d\n' % (i, b))
                if idegree == 0:
                    for l in range(h_sample_rate):
                        result_set.add('%d\t%d\n' % (i, j))
                else:
                    for l in range(h_sample_rate):
                        a = random.sample(neighbor_set_list[i-1][0], 1)[0] + 1
                        result_set.add('%d\t%d\n' % (a, j))
            result_list = [line for line in result_set]
            target_file.writelines(result_list)
        print('\thop2 positive edge count: %d' % len(result_set))
    print('get_hop2_link completed\n')

def get_tdata_with_lable(positive_data_file_path, negative_data_file_path, target_data_file_path):
    print('starting get_tdata_with_lable')

    with open(positive_data_file_path, 'r') as p_file, open(negative_data_file_path, 'r') as n_file, open(target_data_file_path, 'w') as target_file:
        p_list = p_file.readlines()
        n_list = n_file.readlines()
        pl_list = [line[0:-1]+'\t1\n' for line in p_list]
        nl_list = [line[0:-1]+'\t0\n' for line in n_list]
        result_list = pl_list + nl_list
        random.shuffle(result_list)
        target_file.writelines(result_list)
        print('\tcompleted data size: %d'%len(result_list))

    print('get_tdata_with_lable completed\n')

def tsv_to_csv(source_file_path, target_file_path):
    print('starting tsv_to_csv')

    with open(source_file_path,'r') as source_file, open(target_file_path,'w') as target_file:
        source_list = source_file.readlines()
        target_list = [','.join(line.split('\t')) for line in source_list]
        target_file.writelines(target_list)

    print('tsv_to_csv completed\n')

def to_tsv(source_file_path, target_file_path):
    result_list = []
    with open(source_file_path, 'r') as source_file:
        for line in source_file:
            newline = line[0:-2].replace(' ', '\t') + '\n'
            result_list.append(newline)
    with open(target_file_path, 'w') as target_file:
        target_file.writelines(result_list)

def gen_random_data(target_file_path, edge_num, node_num):
    with open(target_file_path, 'w') as target_file:
        edge_set = set()
        while(len(edge_set)<edge_num):
            i = random.randint(1, node_num)
            j = random.randint(1, node_num)
            line = '%d\t%d\n' % (i, j)
            if i != j and line not in edge_set:
                edge_set.add(line)
        result_list = [line for line in edge_set]
        target_file.writelines(result_list)

def store_obj(obj,file_path):
    with open(file_path,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_path):
    if not os.path.exists(file_path):
        return -1
    with open(file_path,'rb') as f:
        return pickle.load(f)

def divide_data_e2e(params):
    path_list = params['source_file_path'].split('/')
    dir = '/'.join(path_list[0:-1]) + '/'
    data_name = path_list[-1].split('_')[0]
    version = params['version']
    node_num = get_max_node_num(params['source_file_path'])
    if params['new_divide']:
        randomly_divide_data(params['source_file_path'],
                             [dir + '%s_train_positive_data_v%d' % (data_name, version),
                              dir + '%s_test_positive_data_v%d' % (data_name, version)],
                             [params['tt_rate'], 1])
    if params['new_tt_data']:
        sample_negative_data(dir + '%s_train_positive_data_v%d' % (data_name, version),
                             dir + '%s_train_negative_data_v%d' % (data_name, version),
                             params['train_np_rate'],
                             node_num,
                             [])
        sample_negative_data(dir + '%s_test_positive_data_v%d' % (data_name, version),
                             dir + '%s_test_negative_data_v%d' % (data_name, version),
                             params['test_np_rate'],
                             node_num,
                             [dir + '%s_train_positive_data_v%d' % (data_name, version),
                              dir + '%s_train_negative_data_v%d' % (data_name, version)])
        get_tdata_with_lable(dir + '%s_train_positive_data_v%d' % (data_name, version),
                             dir + '%s_train_negative_data_v%d' % (data_name, version),
                             dir + '%s_train_data_v%d' % (data_name, version))
        get_tdata_with_lable(dir + '%s_test_positive_data_v%d' % (data_name, version),
                             dir + '%s_test_negative_data_v%d' % (data_name, version),
                             dir + '%s_test_data_v%d' % (data_name, version))

    if params['get_neighbor_set']:
        get_neighbor_set(dir + '%s_train_positive_data_v%d' % (data_name, version),
                         dir + '%s_train_neighbor_set_list_v%d' % (data_name, version),
                         node_num)

    if params['get_katz_matrix']:
        get_katz_matrix(dir + '%s_train_positive_data_v%d' % (data_name, version),
                         dir + '%s_train_katz_matrix_v%d' % (data_name, version),
                         node_num,
                         0.9,
                         params['exact_katz'])

    if params['get_rwr_matrix']:
        get_rwr_matrix(dir + '%s_train_positive_data_v%d' % (data_name, version),
                       dir + '%s_train_neighbor_set_list_v%d' % (data_name, version),
                       dir + '%s_train_rwr_matrix_v%d' % (data_name, version),
                       node_num,
                       0.1,
                       params['exact_rwr'])

    if params['get_hop2_data']:
        get_hop2_link(dir + '%s_train_positive_data_v%d' % (data_name, version),
                      dir + '%s_hop2_train_positive_data_v%d' % (data_name, version),
                      dir + '%s_train_neighbor_set_list_v%d' % (data_name, version),
                      params['random_p'], 2)
        # sample_negative_data(dir + '%s_hop2_train_positive_data_v%d' % (data_name, version),
        #                      dir + '%s_hop2_train_negative_data_v%d' % (data_name, version),
        #                      params['hop2_np_rate'],
        #                      node_num,
        #                      [])
        # get_tdata_with_lable(dir + '%s_hop2_train_positive_data_v%d' % (data_name, version),
        #                      dir + '%s_hop2_train_negative_data_v%d' % (data_name, version),
        #                      dir + '%s_hop2_train_data_v%d' % (data_name, version))

if __name__ == '__main__':
    # gen_random_data('../../data/test1/random_data', 50000, 5000)

    divide_data_e2e({'source_file_path': '../../data/test1/openflights_data',
                     'version': 1,
                     'tt_rate': 4,
                     'train_np_rate': 160,
                     'test_np_rate': 160,
                     'new_divide': True,
                     'new_tt_data': True,
                     'get_neighbor_set': True,
                     'get_katz_matrix': True,
                     'exact_katz': True,
                     'get_rwr_matrix': True,
                     'exact_rwr': True,
                     'get_hop2_data': True,
                     'random_p': False,
                     })