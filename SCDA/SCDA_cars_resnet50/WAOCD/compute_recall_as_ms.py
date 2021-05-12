
import numpy as np
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from os.path import join
import os
import csv

# PARA=[
#         'cars_resnet50_best.pth'
#       ]
#
# PARA=[
#         'resnet50-19c8e357.pth'
#       ]
# PARA = [
#     'resnet101-5d3b4d8f.pth'
# ]
# PARA = [
#     'resnet152-b121ed2d.pth'
# ]

# PARA=[
#     # 'checkpoint_2800_0.5684.pth',
#     # 'checkpoint_2999_0.5039.pth',
#
#     # 'checkpoint_3800_9.1503_arc.pth',
#     # 'checkpoint_3999_8.3020_arc.pth',
#     #
#     'checkpoint_3800_8.1392_cos.pth',
#     'checkpoint_3999_8.7649_cos.pth',
# ]

# PARA=[
#     # 'checkpoint_2800_0.4020.pth',
#     'checkpoint_2999_0.4325.pth',
#
#     # 'checkpoint_3800_7.6700_arc.pth',
#     # 'checkpoint_3999_7.3369_arc.pth',
#     #
#     # 'checkpoint_3800_9.1800_cos.pth',
#     # 'checkpoint_3999_8.3807_cos.pth',
#
# ]
# PARA=[
#
#         'checkpoint_95_0.9858.pth',
#         'checkpoint_100_0.9939.pth',
#         'checkpoint_105_0.9963.pth',
#
# ]
# PARA=[
#
#         # 'checkpoint_95_0.9858.pth',
#         'checkpoint_100_0.9939.pth',
#         'checkpoint_105_0.9965.pth',
#
# ]
# PARA=[
#     # 'checkpoint_2800_0.2730.pth',
#     'checkpoint_2999_0.3844.pth',
#     #
#     # 'checkpoint_3800_9.1503_arc.pth',
#     # 'checkpoint_3999_8.3020_arc.pth',
#     #
#     # 'checkpoint_3800_9.1800_cos.pth',
#     # 'checkpoint_3999_8.7649_cos.pth',
#
# ]
# PARA=[
#     'checkpoint_2800_0.2730.pth',
#     'checkpoint_2999_0.3844.pth',
#
#     'checkpoint_3800_8.8681_arc.pth',
#     'checkpoint_3999_7.8027_arc.pth',
#     #
#     'checkpoint_3800_8.1616_cos.pth',
#     'checkpoint_3999_8.9991_cos.pth',
#
# ]
# PARA=[
#     # 'checkpoint_2800_0.5763.pth',
#     # 'checkpoint_2999_0.6607.pth',
#     #
#     # 'checkpoint_3800_9.0727_arc.pth',
#     # 'checkpoint_3999_6.9830_arc.pth',
#
#     'checkpoint_3800_9.2263_cos.pth',
#     'checkpoint_3999_7.1840_cos.pth',
#
# ]
PARA=[
    # 'checkpoint_2800_0.5620.pth',
    # 'checkpoint_2999_0.5152.pth',
    #
    # 'checkpoint_3800_7.2521_arc.pth',
    # 'checkpoint_3999_7.3297_arc.pth',
    #
    # 'checkpoint_3999_9.6755_arc.pth',
    'checkpoint_3999_10.065_arc.pth',
    # 'checkpoint_3800_9.8567_cos.pth',
    # 'checkpoint_3999_8.4847_cos.pth',

]


# PARA=[
#
#         'checkpoint_95_0.9834.pth',
#         'checkpoint_100_0.9944.pth',
#         'checkpoint_105_0.9951.pth',
#         'checkpoint_110_0.9963.pth',
#
# ]

# PARA=[
#     'checkpoint_2800_0.4723.pth',
#     'checkpoint_2999_0.5073.pth',
#     #
#     'checkpoint_3800_8.1095_arc.pth',
#     'checkpoint_3999_7.0857_arc.pth',
#
#     'checkpoint_3800_8.5838_cos.pth',
#     'checkpoint_3999_8.0785_cos.pth',
#
# ]
# PARA=[
#
#         # 'checkpoint_95_0.9872.pth',
#         'checkpoint_100_0.9934.pth',
#         'checkpoint_105_0.9952.pth',
#         'checkpoint_110_0.9945.pth',
#
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



class RetMetric(object):
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
        # print(np.min(self.sim_mat))

    def recall_k(self, k=1):
        m = len(self.sim_mat)

        match_counter = 0

        for i in range(m):

            pos_sim = self.sim_mat[i][self.gallery_labels == self.query_labels[i]]
            neg_sim = self.sim_mat[i][self.gallery_labels != self.query_labels[i]]

            thresh = np.sort(pos_sim)[-2] if self.is_equal_query else np.max(pos_sim)
            # if i==5000:
            #     print('###################')
            #     print(np.sort(pos_sim)[-12:-2])
            #     print(np.sort(pos_sim)[0:10])
            #     print(np.max(neg_sim))
            #     print(np.min(neg_sim))
            #     print('$$$$$$$$$$$$$$$$$$$')
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
