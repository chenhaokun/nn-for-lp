#coding: utf-8
from graph_interpreter import *
from data_divider import *
import time

if __name__ == '__main__':

    divide_data_e2e({'source_file_path': '../data/openflights_data',
                     'version': 1,
                     'tt_rate': 4,
                     'train_np_rate': 20,
                     'test_np_rate': 20,
                     'new_divide': True,
                     'new_tt_data': True,
                     'get_neighbor_set': True,
                     'get_katz_matrix': True,
                     'get_rwr_matrix': True,
                     'get_hop2_data': True,
                     'random_p': False})
    
    base_exp({'source_file_path': '../data/openflights_data',
              'version': 1,
              'train_np_rate': 20,
              'baseline_set': set(['cn','aa','ra','katz','rwr','mf']),
              'pnn1': {'learning_rate': 1e-2, 'beta': 1e-5, 'round': 40},
              'pnn2': {'learning_rate1': 1e-4, 'learning_rate2': 1e-4, 'beta1': 1e-5, 'beta2': 1e-5,
              'hop2_np_rate': 10, 'round': 25}})
