#coding: utf-8
from graph_interpreter import *
from divide_data_e2e import *
import time

if __name__ == '__main__':

    # divide_data_e2e({'source_file_path': '../../data/test1/cora_data',
    #                  'version': 33,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 20,
    #                  'test_np_rate': 20,
    #                  'new_divide': True,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/cora_data',
    #           'version': 33,
    #           'train_np_rate': 20,
    #           'baseline_set': set([]),
    #           'pnn1_test': {'learning_rate': 1e-2, 'beta': 1e-5, 'round': 40},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 1e-4, 'learning_rate2': 1e-4, 'beta1': 1e-5, 'beta2': 1e-5,
    #                         'hop2_np_rate': 10, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})

    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                                   'version': 35,
    #                                   'tt_rate': 4,
    #                                   'train_np_rate': 20,
    #                                   'test_np_rate': 20,
    #                                   'new_divide': True,
    #                                   'new_tt_data': False,
    #                                   'get_neighbor_set': False,
    #                                   'get_katz_matrix': False,
    #                                   'exact_katz': True,
    #                                   'get_rwr_matrix': False,
    #                                   'exact_rwr': True,
    #                                   'get_hop2_data': False,
    #                                   'random_p': False})

    # noise_exp({'source_file_path': '../../data/test1/openflights_data',
    #            'version': 35,
    #            'new_data': False,
    #            'train_np_rate': 20,
    #            'test_np_rate': 20,
    #            'random_p': False,
    #            'ns_rate_list': [0.25, 0.50, 0.75, 1.00]})

    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                   'version': 1,
    #                   'tt_rate': 4,
    #                   'train_np_rate': 160,
    #                   'test_np_rate': 160,
    #                   'new_divide': True,
    #                   'new_tt_data': True,
    #                   'get_neighbor_set': True,
    #                   'get_katz_matrix': False,
    #                   'exact_katz': False,
    #                   'get_rwr_matrix': False,
    #                   'exact_rwr': False,
    #                   'get_hop2_data': True,
    #                   'random_p': False})
    #
    # params_exp({'source_file_path': '../../data/test1/small_data',
    #             'version': 1,
    #             'embedding_size_list': [4],
    #             'h_size_list': [[20], [20, 20], [20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20, 20]],
    #             'rewrite': True
    #             })
    #
    # params_exp({'source_file_path': '../../data/test1/small_data',
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




    # show_params_test_result({'source_file_path': '../../data/test1/small_data', 'version': 1})
    # show_params_exp_result('embedding_size')
    # base_exp({'source_file_path': '../../data/test1/openflights_data',
    #           'version': 37,
    #           'train_np_rate': 20,
    #           'baseline_set': set([]),
    #           'pnn1_test': {},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {},
    #           'pnn2': False,
    #           'store_test_result': False,
    #           'show_auc_curve': {},#'x_min':1, 'x_max':22, 'x_step':3, 'y_min':0.80, 'y_max':0.969#v:20
    #           'show_embedding_distribution': {'x_min': 1, 'x_max': 22, 'x_step': 3, 'y_min': 0.80, 'y_max': 0.969}}) #'x_min': 1, 'x_max': 22, 'x_step': 3, 'y_min': 0.80, 'y_max': 0.969}})
    # show_completeness_test_result({'source_file_path': '../../data/test1/openflights_data', 'version': 31})
    show_noise_exp_result('openflights5')

    # data_name = 'enron'
    #
    # print 'start %s at:' % data_name
    # print time.localtime(time.time())
    #
    # divide_data_e2e({'source_file_path': '../../data/test1/%s_data' % data_name,
    #                  'version': 39,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 20,
    #                  'test_np_rate': 20,
    #                  'new_divide': True,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # print 'end data dividing of %s at:' % data_name
    # print time.localtime(time.time())
    #
    # base_exp({'source_file_path': '../../data/test1/%s_data' % data_name,
    #           'version': 39,
    #           'train_np_rate': 20,
    #           'baseline_set': set(['']),
    #           'pnn1_test': {'learning_rate': 1e-2, 'beta': 2e-5, 'round': 10},#'learning_rate': 1e-2, 'beta': 2e-5, 'round': 40},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})
    #
    # print 'end pnn1 of %s at:' % data_name
    # print time.localtime(time.time())
    #
    # base_exp({'source_file_path': '../../data/test1/%s_data' % data_name,
    #           'version': 39,
    #           'train_np_rate': 20,
    #           'baseline_set': set(['']),
    #           'pnn1_test': {},
    #           # 'learning_rate': 1e-2, 'beta': 2e-5, 'round': 40},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 1e-2, 'learning_rate2': 1e-2, 'beta1': 2e-5, 'beta2': 2e-5,
    #                         'hop2_np_rate': 4, 'round': 10},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})
    #
    # print 'end pnn2 of %s at:' % data_name
    # print time.localtime(time.time())

    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                  'version': 1,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 80,
    #                  'test_np_rate': 80,
    #                  'new_divide': True,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/small_data',
    #           'version': 1,
    #           'train_np_rate': 80,
    #           'baseline_set': set(['mf']),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4, 'hop2_np_rate': 80, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})
    #
    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                  'version': 2,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 160,
    #                  'test_np_rate': 160,
    #                  'new_divide': True,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/small_data',
    #           'version': 2,
    #           'train_np_rate': 160,
    #           'baseline_set': set(['mf']),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4,
    #                         'hop2_np_rate': 160, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})
    #
    # divide_data_e2e({'source_file_path': '../../data/test1/enron_data',
    #                 'version': 1,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 160,
    #                  'test_np_rate': 160,
    #                  'new_divide': True,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': True,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': False,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/enron_data',
    #           'version': 1,
    #           'train_np_rate': 160,
    #           'baseline_set': set(['rwr']),
    #           'pnn1_test': {},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})

    # completeness_exp({'source_file_path': '../../data/test1/openflights_data',
    #                   'version': 1,
    #                   'train_np_rate': 160,
    #                   'hop2_np_rate': 16,
    #                   'h_sample_rate': 5,
    #                   'training_set_num': 4,
    #                   'divide_data': True})
    # completeness_exp_v2({'source_file_path': '../../data/test1/openflights_data',
    #                   'version': 31,
    #                   'train_np_rate': 20,
    #                   'hop2_np_rate': 2,
    #                   'h_sample_rate': 10,
    #                   'divide_list': [0.2,0.4,0.6,0.8],#0.04,0.06,0.08,0.10,
    #                   'divide_data': False})
    # completeness_exp_v2({'source_file_path': '../../data/test1/small_data',
    #                      'version': 31,
    #                      'train_np_rate': 160,
    #                      'hop2_np_rate': 16,
    #                      'h_sample_rate': 5,
    #                      'divide_list': [0.5, 0.6, 0.7, 0.8],
    #                      'divide_data': True})



    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                  'version': 10,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 10,
    #                  'test_np_rate': 10,
    #                  'new_divide': False,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/small_data',
    #           'version': 10,
    #           'train_np_rate': 10,
    #           'baseline_set': set([]),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4,
    #                         'hop2_np_rate': 10, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})
    #
    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                  'version': 11,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 20,
    #                  'test_np_rate': 20,
    #                  'new_divide': False,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/small_data',
    #           'version': 11,
    #           'train_np_rate': 20,
    #           'baseline_set': set([]),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4,
    #                         'hop2_np_rate': 20, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})
    #
    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                  'version': 12,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 40,
    #                  'test_np_rate': 40,
    #                  'new_divide': False,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/small_data',
    #           'version': 12,
    #           'train_np_rate': 40,
    #           'baseline_set': set([]),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4,
    #                         'hop2_np_rate': 40, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})
    #

    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                  'version': 13,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 80,
    #                  'test_np_rate': 80,
    #                  'new_divide': False,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/small_data',
    #           'version': 13,
    #           'train_np_rate': 80,
    #           'baseline_set': set([]),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4,
    #                         'hop2_np_rate': 80, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})
    #
    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                  'version': 13,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 80,
    #                  'test_np_rate': 80,
    #                  'new_divide': False,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/small_data',
    #           'version': 13,
    #           'train_np_rate': 80,
    #           'baseline_set': set([]),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4,
    #                         'hop2_np_rate': 80, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})
    #
    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                  'version': 14,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 160,
    #                  'test_np_rate': 160,
    #                  'new_divide': False,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/small_data',
    #           'version': 14,
    #           'train_np_rate': 160,
    #           'baseline_set': set([]),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4,
    #                         'hop2_np_rate': 160, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})
    #
    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                  'version': 14,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 160,
    #                  'test_np_rate': 160,
    #                  'new_divide': False,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/small_data',
    #           'version': 14,
    #           'train_np_rate': 160,
    #           'baseline_set': set([]),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4,
    #                         'hop2_np_rate': 160, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})
    #
    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                  'version': 15,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 320,
    #                  'test_np_rate': 320,
    #                  'new_divide': False,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/small_data',
    #           'version': 15,
    #           'train_np_rate': 320,
    #           'baseline_set': set([]),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4,
    #                         'hop2_np_rate': 160, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})
    #
    #
    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                  'version': 16,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 640,
    #                  'test_np_rate': 640,
    #                  'new_divide': False,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/small_data',
    #           'version': 16,
    #           'train_np_rate': 640,
    #           'baseline_set': set([]),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4,
    #                         'hop2_np_rate': 160, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})
    #
    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                  'version': 15,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 320,
    #                  'test_np_rate': 320,
    #                  'new_divide': False,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/small_data',
    #           'version': 15,
    #           'train_np_rate': 320,
    #           'baseline_set': set([]),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4,
    #                         'hop2_np_rate': 160, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})
    #
    # divide_data_e2e({'source_file_path': '../../data/test1/small_data',
    #                  'version': 16,
    #                  'tt_rate': 4,
    #                  'train_np_rate': 640,
    #                  'test_np_rate': 640,
    #                  'new_divide': False,
    #                  'new_tt_data': True,
    #                  'get_neighbor_set': True,
    #                  'get_katz_matrix': False,
    #                  'exact_katz': True,
    #                  'get_rwr_matrix': False,
    #                  'exact_rwr': True,
    #                  'get_hop2_data': True,
    #                  'random_p': False})
    #
    # base_exp({'source_file_path': '../../data/test1/small_data',
    #           'version': 16,
    #           'train_np_rate': 640,
    #           'baseline_set': set([]),
    #           'pnn1_test': {'round': 25},
    #           'pnn1': False,
    #           'fixed_emb_pnn2': False,
    #           'pnn2_test': {'learning_rate1': 4e-3, 'learning_rate2': 4e-3, 'beta1': 4e-4, 'beta2': 4e-4,
    #                         'hop2_np_rate': 160, 'round': 25},
    #           'pnn2': False,
    #           'store_test_result': True,
    #           'show_auc_curve': {},
    #           'show_embedding_distribution': {}})



