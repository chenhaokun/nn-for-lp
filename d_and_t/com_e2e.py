#coding: utf-8
from graph_interpreter import *
from divide_data_e2e import *

if __name__ == '__main__':

    # params_exp({'source_file_path': '../../data/test1/openflights_data',
    #             'version': 1,
    #             'embedding_size_list': [20],
    #             'h_size_list': [[20], [20, 20], [20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20]],
    #             'rewrite': True
    #             })
    #
    # params_exp({'source_file_path': '../../data/test1/openflights_data',
    #             'version': 1,
    #             'embedding_size_list': [5,10,15,20,25,30,35,40],
    #             'h_size_list': [[20, 20, 20]],
    #             'rewrite': False
    #             })
    #
    # params_exp({'source_file_path': '../../data/test1/openflights_data',
    #             'version': 1,
    #             'embedding_size_list': [5, 10, 15, 20, 25, 30, 35, 40],
    #             'h_size_list': [[20, 20, 20, 20]],
    #             'rewrite': False
    #             })
    #
    # params_exp({'source_file_path': '../../data/test1/cora_data',
    #             'version': 1,
    #             'embedding_size_list': [20],
    #             'h_size_list': [[20], [20, 20], [20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20]],
    #             'rewrite': False
    #             })




    # show_params_test_result({'source_file_path': '../../data/test1/openflights_data', 'version': 1})
    # base_exp({'source_file_path': '../../data/test1/cora_data',
    #           'version': 1,
    #           'train_np_rate': 160,
    #           'baseline_set': set([]),
    #           'pnn1_test': {},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': True})






    divide_data_e2e({'source_file_path': '../../data/test1/openflights_data',
                     'version': 1,
                     'tt_rate': 4,
                     'train_np_rate': 160,
                     'test_np_rate': 160,
                     'new_divide': False,
                     'new_tt_data': False,
                     'get_neighbor_set': False,
                     'get_katz_matrix': False,
                     'exact_katz': True,
                     'get_rwr_matrix': True,
                     'exact_rwr': True,
                     'get_hop2_data': False,
                     'random_p': False})

    base_exp({'source_file_path': '../../data/test1/openflights_data',
              'version': 1,
              'train_np_rate': 160,
              'baseline_set': set(['rwr']),
              'pnn1_test': {},
              'pnn1': False,
              'fixed_emb_pnn2': False,
              'pnn2_test': {},
              'pnn2': False,
              'store_test_result': True,
              'show_auc_curve': False})

    divide_data_e2e({'source_file_path': '../../data/test1/cora_data',
                     'version': 1,
                     'tt_rate': 4,
                     'train_np_rate': 160,
                     'test_np_rate': 160,
                     'new_divide': False,
                     'new_tt_data': False,
                     'get_neighbor_set': False,
                     'get_katz_matrix': False,
                     'exact_katz': True,
                     'get_rwr_matrix': True,
                     'exact_rwr': True,
                     'get_hop2_data': False,
                     'random_p': False})

    base_exp({'source_file_path': '../../data/test1/cora_data',
              'version': 1,
              'train_np_rate': 160,
              'baseline_set': set(['rwr']),
              'pnn1_test': {},
              'pnn1': False,
              'fixed_emb_pnn2': False,
              'pnn2_test': {},
              'pnn2': False,
              'store_test_result': True,
              'show_auc_curve': False})

    divide_data_e2e({'source_file_path': '../../data/test1/small_data',
                     'version': 1,
                     'tt_rate': 4,
                     'train_np_rate': 160,
                     'test_np_rate': 160,
                     'new_divide': False,
                     'new_tt_data': False,
                     'get_neighbor_set': False,
                     'get_katz_matrix': False,
                     'exact_katz': True,
                     'get_rwr_matrix': True,
                     'exact_rwr': True,
                     'get_hop2_data': False,
                     'random_p': False})

    base_exp({'source_file_path': '../../data/test1/small_data',
              'version': 1,
              'train_np_rate': 160,
              'baseline_set': set(['rwr']),
              'pnn1_test': {},
              'pnn1': False,
              'fixed_emb_pnn2': False,
              'pnn2_test': {},
              'pnn2': False,
              'store_test_result': True,
              'show_auc_curve': False})

    divide_data_e2e({'source_file_path': '../../data/test1/enron_data',
                     'version': 1,
                     'tt_rate': 4,
                     'train_np_rate': 160,
                     'test_np_rate': 160,
                     'new_divide': True,
                     'new_tt_data': True,
                     'get_neighbor_set': True,
                     'get_katz_matrix': True,
                     'exact_katz': False,
                     'get_rwr_matrix': False,
                     'exact_rwr': True,
                     'get_hop2_data': True,
                     'random_p': False})

    base_exp({'source_file_path': '../../data/test1/enron_data',
              'version': 1,
              'train_np_rate': 160,
              'baseline_set': set(['cn', 'aa', 'ra', 'katz', 'mf']),
              'pnn1_test': {'round': 25},
              'pnn1': False,
              'fixed_emb_pnn2': False,
              'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4,
                            'hop2_np_rate': 160, 'round': 25},
              'pnn2': False,
              'store_test_result': True,
              'show_auc_curve': False})