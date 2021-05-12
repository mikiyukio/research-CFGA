
import numpy as np
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from os.path import join
import os
import csv

# PARA=[
#         'cub_resnet50_best.pth'
#       ]
# PARA=[
#     # 'checkpoint_2999_0.2883.pth',
#     #   'checkpoint_2800_6.8010_arc.pth',
#     #   'checkpoint_2999_7.9913_arc.pth',
#     #   'checkpoint_2800_9.1686_cos.pth',
#     #   'checkpoint_2999_8.0910_cos.pth',
#       'checkpoint_2800_0.3830.pth',
#       'checkpoint_2999_0.2896.pth',
#
# ]
# PARA=[
#       'checkpoint_2800_10.976_cos.pth',
#       'checkpoint_2999_7.5176_cos.pth',
#       # 'checkpoint_2800_0.3830.pth',
#       # 'checkpoint_2999_0.2896.pth',
# ]
# PARA=[
#       'checkpoint_2800_7.8380_arc.pth',
#       'checkpoint_2999_7.0835_arc.pth',
#       # 'checkpoint_2800_0.3830.pth',
#       # 'checkpoint_2999_0.2896.pth',
# ]
# PARA=[
#     # 'checkpoint_19_0.7206.pth',
#
#     # 'checkpoint_95_0.9933.pth',
#     'checkpoint_100_0.9996.pth',
#     'checkpoint_100_1.0.pth',
#     # 'checkpoint_20_0.7326.pth',
#
# ]
# PARA=[
#       'checkpoint_2800_0.2908.pth',
#       'checkpoint_2999_0.4089.pth',
#       'checkpoint_2800_7.9556_arc.pth',
#       'checkpoint_2999_7.3573_arc.pth',
#       'checkpoint_2800_10.519_cos.pth',
#       'checkpoint_2999_8.7425_cos.pth',
#     #   'checkpoint_2800_0.2575.pth',
#     #   'checkpoint_2999_0.2884.pth',
#
# ]
PARA=[
        # 'resnet50-19c8e357.pth',
    # 'checkpoint_2999_10.273_cos.pth',
        # 'resnet101-5d3b4d8f.pth',
        # 'resnet152-b121ed2d.pth'
        'cub_resnet50_best.pth'
      ]

# PARA=[
#     # 'checkpoint_19_0.7206.pth',
#
#     'checkpoint_95_0.9933.pth',
#     'checkpoint_100_0.9986.pth',
#     'checkpoint_105_1.0.pth',
#     # 'checkpoint_20_0.7326.pth',
#
# ]
# PARA=[
#       'checkpoint_2800_7.7078_arc.pth',
#       'checkpoint_2999_9.2809_arc.pth',
#       'checkpoint_2800_8.1657_cos.pth',
#       'checkpoint_2999_7.2830_cos.pth',
# ]
# PARA=['checkpoint_2800_0.1710.pth',
#       'checkpoint_2999_0.3672.pth']
# PARA=['checkpoint_2800_0.4535.pth',
#       'checkpoint_2999_0.3527.pth']
# PARA=[
#       'checkpoint_2800_0.4841.pth',
#       'checkpoint_2999_0.3712.pth',
#       'checkpoint_2800_9.0264_arc.pth',
#       'checkpoint_2999_6.4793_arc.pth',
#       'checkpoint_2800_7.3284_cos.pth',
#       'checkpoint_2999_7.2485_cos.pth',
#     #   'checkpoint_2800_0.2575.pth',
#     #   'checkpoint_2999_0.2884.pth',
#
# ]

# PARA=[
#     'checkpoint_2800_0.4553.pth',
#     'checkpoint_2999_0.3860.pth',
#     'checkpoint_2800_0.4602.pth',
#     'checkpoint_2999_0.4015.pth',
#     'checkpoint_2800_0.5468.pth',
#     'checkpoint_2999_0.5321.pth',
#
#     #
#     # 'checkpoint_3000_0.2741.pth',
#     # 'checkpoint_3200_0.3272.pth',
#     # 'checkpoint_3399_0.2182.pth',
#
#
# ]

# PARA=[
#     # 'checkpoint_19_0.7206.pth',
#
#     'checkpoint_95_0.9947.pth',
#     'checkpoint_100_0.9993.pth',
#     'checkpoint_105_1.0.pth',
#     'checkpoint_110_1.0.pth',
#
#     # 'checkpoint_20_0.7326.pth',
#
# ]

# PARA=[
#       'checkpoint_2800_0.4899.pth',
#       'checkpoint_2999_0.1897.pth',
#       'checkpoint_2800_7.8492_arc.pth',
#       'checkpoint_2999_9.6859_arc.pth',
#       'checkpoint_2800_8.8039_cos.pth',
#       'checkpoint_2999_6.3490_cos.pth',
#     #   'checkpoint_2800_0.2575.pth',
#     #   'checkpoint_2999_0.2884.pth',
#
# ]
# PARA=[
#       # 'checkpoint_2800_7.8380_arc.pth',
#       # 'checkpoint_2999_7.0835_arc.pth',
#       # 'checkpoint_2800_0.2779.pth',
#       # 'checkpoint_2999_0.3576.pth',
#       # 'checkpoint_2800_7.3311_arc.pth',
#       # 'checkpoint_2999_6.8704_arc.pth',
#       'checkpoint_2800_11.913_cos.pth',
#       'checkpoint_2999_8.9631_cos.pth',
# ]

# PARA=[
#       # 'checkpoint_2800_7.8380_arc.pth',
#       # 'checkpoint_2999_7.0835_arc.pth',
#       'checkpoint_2800_0.3296.pth',
#       'checkpoint_2999_0.4563.pth',
#       # 'checkpoint_2800_7.3311_arc.pth',
#       # 'checkpoint_2999_6.8704_arc.pth',
#       # 'checkpoint_2800_11.913_cos.pth',
#       # 'checkpoint_2999_8.9631_cos.pth',
# ]
# PARA=[
#     # 'checkpoint_19_0.7206.pth',
#
#     # 'checkpoint_95_0.9965.pth',
#     'checkpoint_100_0.9994.pth',
#     'checkpoint_105_1.0.pth',
#     'checkpoint_110_1.0.pth',
#
#     # 'checkpoint_20_0.7326.pth',
#
# ]

# PARA=[
#       # 'checkpoint_2800_0.2744.pth',
#       # 'checkpoint_2999_0.3422.pth',
#       # 'checkpoint_2800_7.8492_arc.pth',
#       'checkpoint_2999_9.6859_arc.pth',
#       'checkpoint_2800_8.8039_cos.pth',
#       # 'checkpoint_2999_6.3490_cos.pth',
#     #   'checkpoint_2800_0.2575.pth',
#     #   'checkpoint_2999_0.2884.pth',
#
# ]



# PARA=[
#     # 'checkpoint_19_0.7206.pth',
#
#     # 'checkpoint_95_0.9965.pth',
#     # 'checkpoint_100_0.9981.pth',
#     # 'checkpoint_100_0.9993.pth',
#     # 'checkpoint_100_0.9994.pth',
#     # 'checkpoint_100_0.9996.pth',
#     # 'checkpoint_100_0.9998.pth',
#     # 'checkpoint_100_1.0.pth',
#     # 'checkpoint_105_0.9996.pth',
#     # 'checkpoint_105_0.9998.pth',
#     # 'checkpoint_105_1.0.pth',
#     'checkpoint_105_1.0000.pth',
#     'checkpoint_105_1.00000.pth',
#
#     # 'checkpoint_110_1.0.pth',
#
#     # 'checkpoint_20_0.7326.pth',
#
# ]

# PARA=[
#       # 'checkpoint_2800_7.8380_arc.pth',
#       # 'checkpoint_2999_7.0835_arc.pth',
#       'checkpoint_2800_0.3830.pth',
#       'checkpoint_2999_0.2896.pth',
#       'checkpoint_2999_0.4563.pth',
#       'checkpoint_2800_7.3311_arc.pth',
#       'checkpoint_2999_6.8704_arc.pth',
#       'checkpoint_2800_11.913_cos.pth',
#       'checkpoint_2999_8.9631_cos.pth',
# ]
#
# PARA=[
#       # 'checkpoint_2800_7.8380_arc.pth',
#       # 'checkpoint_2999_7.0835_arc.pth',
#       'checkpoint_2800_0.3830.pth',
#       'checkpoint_2999_0.2896.pth',
#       'checkpoint_2800_7.3311_arc.pth',
#       'checkpoint_2999_6.8704_arc.pth',
#       'checkpoint_2800_11.913_cos.pth',
#       'checkpoint_2999_8.9631_cos.pth',
# ]

# PARA=[

      # 'checkpoint_2800_7.8380_arc.pth',
      # 'checkpoint_2999_7.0835_arc.pth',
      # 'checkpoint_2800_0.3830.pth',
      # 'checkpoint_2999_0.2896.pth',
      # 'checkpoint_2999_0.4563.pth',
    # 'checkpoint_2800_6.9309_arc.pth',
    # 'checkpoint_2800_7.3311_arc.pth',
    # 'checkpoint_2800_10.324_arc.pth',
    # 'checkpoint_2999_6.8704_arc.pth',
    # 'checkpoint_2999_7.5413_arc.pth',
    # 'checkpoint_2800_11.913_cos.pth',
    #   'checkpoint_2999_8.9631_cos.pth',
    # 'checkpoint_2800_7.3088_arc.pth',
    #   'checkpoint_2999_5.2132_arc.pth',
# 'checkpoint_2999_7.7312_arc.pth',
#     'checkpoint_2999_8.9352_arc.pth',
#     'checkpoint_2999_10.575_arc.pth',
#
#     'checkpoint_2999_9.3775_arc.pth',
#     'checkpoint_2999_11.055_arc.pth',
# 'checkpoint_2999_8.9676_arc.pth',
    # 'checkpoint_2800_7.3758_arc.pth',
    # 'checkpoint_2999_8.4362_arc.pth',
# ]
#
# PARA=[
#       # 'checkpoint_2800_0.3183.pth',
#       'checkpoint_2999_0.3919.pth',
#       'checkpoint_2800_0.1907.pth',
#       'checkpoint_2999_0.4293.pth',
#       # 'checkpoint_2800_7.8492_arc.pth',
#       'checkpoint_2999_9.6859_arc.pth',
#       'checkpoint_2800_8.2702_cos.pth',
#       'checkpoint_2800_8.8039_cos.pth',
#       'checkpoint_2999_8.9177_cos.pth',
#
#     # 'checkpoint_2800_0.3344.pth',
#     # 'checkpoint_2999_0.1836.pth',
#
# ]


# PARA=[
#       # 'checkpoint_2800_0.2744.pth',
#       # 'checkpoint_2999_0.3422.pth',
#       # 'checkpoint_2800_7.8492_arc.pth',
#       'checkpoint_2999_9.6859_arc.pth',
#       'checkpoint_2800_8.8039_cos.pth',
#
# ]
# PARA=[
#       # 'checkpoint_2800_0.3183.pth',
#       # 'checkpoint_2999_0.3919.pth',
#     'checkpoint_2800_0.1907.pth',
#     'checkpoint_2999_0.4293.pth',
#     # 'checkpoint_2800_9.8919_arc.pth',
#     # 'checkpoint_2999_8.9412_arc.pth',
# ]
files_json=[
            'layer4_1_conv1_weight.json',
            'layer4_1_conv2_weight.json','layer4_2_conv2_weight.json',
            'layer4_conv2_weight.json',
            # 'layer4_conv2_weight_R34.json',

    #
            'layer4_1_conv3_weight.json',
            'layer4_2_conv3_weight.json',
            'layer4_2_conv1_weight.json',

            # 'embedding_norm_max_avg.json',
            'embedding_max_avg.json',
            'scda_max_avg_norm.json',
            'scda_norm_max_avg.json',

            ]
results_csv=[
            'layer4_1_conv1_weight.csv',
            'layer4_1_conv2_weight.csv','layer4_2_conv2_weight.csv',
            'layer4_conv2_weight.csv',
            # 'layer4_conv2_weight_R34.csv',

    #
            'layer4_1_conv3_weight.csv',
            'layer4_2_conv3_weight.csv',
            'layer4_2_conv1_weight.csv',
            #
            # 'embedding_norm_max_avg.csv',
            'embedding_max_avg.csv',
            'scda_max_avg_norm.csv',
            'scda_norm_max_avg.csv',
            ]



class RetMetric():
    def __init__(self, feats, labels):

        if len(feats) == 2 and type(feats) == list:
            """
            feats = [gallery_feats, query_feats]
            labels = [gallery_labels, query_labels]
            """
            self.is_equal_query = False

            self.gallery_feats, self.query_feats = feats
            self.gallery_labels, self.query_labels = labels

        else:
            self.is_equal_query = True
            self.gallery_feats = self.query_feats = feats
            self.gallery_labels = self.query_labels = labels

        self.sim_mat = np.matmul(self.query_feats, np.transpose(self.gallery_feats))

    def recall_k(self, k=1):
        m = len(self.sim_mat)

        match_counter = 0

        for i in range(m):
            pos_sim = self.sim_mat[i][self.gallery_labels == self.query_labels[i]]
            neg_sim = self.sim_mat[i][self.gallery_labels != self.query_labels[i]]

            thresh = np.sort(pos_sim)[-2] if self.is_equal_query else np.max(pos_sim)

            if np.sum(neg_sim > thresh) < k:
                match_counter += 1
        return float(match_counter) / m






for index in range(len(PARA)):
    PARA_I = PARA[index].replace('.', '_')  # 避免路径出错
    target_path = join(os.path.abspath(os.path.dirname(os.getcwd())), 'datafile', PARA_I)

    for index_2 in range(len(files_json)):

        filename=join(target_path,files_json[index_2])
        print(filename)
        # f = open(join(os.path.abspath(os.path.dirname(os.getcwd())),'result',results_csv[index_2]),'a+',encoding='utf-8',newline='' "")
        # ff = open(join(os.path.abspath(os.path.dirname(os.getcwd())),'result',results_csv[index_2]),'r',encoding='utf-8',newline='' "")
        # l=len(ff.readlines())
        # ff.close()





        with open(filename) as f_obj:
            final_features=json.load(f_obj)
        # train_data=final_features['train']
        test_data=final_features['test']
        print('ok******************')


        train_paths_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/train_paths.json')
        test_paths_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/test_paths.json')
        train_labels_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/train_labels.json')
        test_labels_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/test_labels.json')
        with open(train_paths_name) as miki:
                train_paths=json.load(miki)
        with open(test_paths_name) as miki:
                test_paths=json.load(miki)
        with open(train_labels_name) as miki:
                train_labels=json.load(miki)
        with open(test_labels_name) as miki:
                test_labels=json.load(miki)


        ########################333
        # dataset_labels=test_labels+train_labels
        dataset_labels=np.array(test_labels)

        #############################
        X = np.array(test_data)
        print(X.shape)

        metric=RetMetric(X,dataset_labels)
        print(metric.recall_k(1))
        print(metric.recall_k(2))
        print(metric.recall_k(4))
        print(metric.recall_k(8))
        print(metric.recall_k(16))
        print(metric.recall_k(32))
