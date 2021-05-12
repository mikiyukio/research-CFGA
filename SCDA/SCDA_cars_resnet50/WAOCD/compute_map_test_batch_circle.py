import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KDTree
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from os.path import join
import os
import csv



#
# PARA=[
#
#     'checkpoint_2800_0.3606.pth',
#     'checkpoint_2800_0.4311.pth',
#     'checkpoint_2800_0.4378.pth',
#     'checkpoint_2800_0.4384.pth',
#     'checkpoint_2800_0.4461.pth',
#     'checkpoint_2800_0.4773.pth',
#     'checkpoint_2800_0.4800.pth',
#     'checkpoint_2800_0.5450.pth',
#     'checkpoint_2800_0.5872.pth',
#
#     'checkpoint_2999_0.4325.pth',
#     'checkpoint_2999_0.4400.pth',
#     'checkpoint_2999_0.4474.pth',
#     'checkpoint_2999_0.4629.pth',
#     'checkpoint_2999_0.4846.pth',
#     'checkpoint_2999_0.5081.pth',
#     'checkpoint_2999_0.5329.pth',
#     'checkpoint_2999_0.5884.pth',
#     'checkpoint_2999_0.6013.pth',
#
# ]



# PARA=[
#
#     'checkpoint_2800_0.3805.pth',
#     'checkpoint_2800_0.4289.pth',
#     'checkpoint_2800_0.4476.pth',
#     'checkpoint_2800_0.4764.pth',
#     'checkpoint_2800_0.5057.pth',
#     # 'checkpoint_2800_0.4773.pth',
#     # 'checkpoint_2800_0.4800.pth',
#     # 'checkpoint_2800_0.5450.pth',
#     # 'checkpoint_59.pth',
#
#     # 'checkpoint_3999_6.5753.pth',
#     # 'checkpoint_3999_6.8874.pth',
#     # 'checkpoint_3999_9.9279.pth',
#     # 'checkpoint_3999_8.8550.pth',
#     # 'checkpoint_3999_10.683.pth',
#
#     # 'checkpoint_7999_3.4940.pth',
#     # 'checkpoint_2999_0.4629.pth',
#     'checkpoint_2999_0.4603.pth',
#     'checkpoint_2999_0.4664.pth',
#     'checkpoint_2999_0.4969.pth',
#     'checkpoint_2999_0.5146.pth',
#     'checkpoint_2999_0.5183.pth',
#
# ]

# PARA=[
#
#     'checkpoint_3800_8.2645.pth',
#     'checkpoint_3800_8.3732.pth',
#     'checkpoint_3800_10.186.pth',
#     'checkpoint_3800_11.445.pth',
#     # 'checkpoint_2800_0.5057.pth',
#     # 'checkpoint_2800_0.4773.pth',
#     # 'checkpoint_2800_0.4800.pth',
#     # 'checkpoint_2800_0.5450.pth',
#     # 'checkpoint_59.pth',
#
#     # 'checkpoint_3999_6.5753.pth',
#     # 'checkpoint_3999_6.8874.pth',
#     # 'checkpoint_3999_9.9279.pth',
#     # 'checkpoint_3999_8.8550.pth',
#     # 'checkpoint_3999_10.683.pth',
#
#     # 'checkpoint_7999_3.4940.pth',
#     # 'checkpoint_2999_0.4629.pth',
#     # 'checkpoint_2999_0.4603.pth',
#     'checkpoint_3999_6.5753.pth',
#     'checkpoint_3999_9.6641.pth',
#     'checkpoint_3999_9.9116.pth',
#     'checkpoint_3999_10.292.pth',
#
# ]

#
# PARA=[
#
#     # 'checkpoint_3800_5.7351.pth',
#     # 'checkpoint_3800_8.2911.pth',
#     # 'checkpoint_3800_9.3602.pth',
#     # 'checkpoint_3800_11.201.pth',
#     # 'checkpoint_2800_0.5057.pth',
#     # 'checkpoint_2800_0.4773.pth',
#     # 'checkpoint_2800_0.4800.pth',
#     # 'checkpoint_2800_0.5450.pth',
#     'cars_resnet50_best.pth',
#
#     # 'checkpoint_3999_6.5753.pth',
#     # 'checkpoint_3999_6.8874.pth',
#     # 'checkpoint_3999_9.9279.pth',
#     # 'checkpoint_3999_8.8550.pth',
#     # 'checkpoint_3999_10.683.pth',
#
#     # 'checkpoint_7999_3.4940.pth',
#     # 'checkpoint_2999_0.4629.pth',
#     # 'checkpoint_2999_0.4603.pth',
#     # 'checkpoint_3999_6.6162.pth',
#     # 'checkpoint_3999_7.0664.pth',
#     # 'checkpoint_3999_7.8628.pth',
#     # 'checkpoint_3999_9.0609.pth',
#
# ]
# PARA=[
#     # 'checkpoint_2800_0.2901.pth',
#     # 'checkpoint_2800_0.3014.pth',
#     # 'checkpoint_2800_0.3096.pth',
#     # 'checkpoint_2800_0.3167.pth',
#     # 'checkpoint_2800_0.3508.pth',
#     # 'checkpoint_2800_0.3725.pth',
#     # 'checkpoint_2800_0.3787.pth',
#     'checkpoint_3800_7.7058.pth',
#     'checkpoint_3800_8.9178.pth',
#     'checkpoint_3800_9.7122.pth',
#
#     'checkpoint_3999_7.8551.pth',
#     'checkpoint_3999_7.9934.pth',
#     'checkpoint_3999_8.1258.pth',
#     'checkpoint_3999_8.6107.pth',
#     'checkpoint_3999_8.7967.pth',
#     'checkpoint_3999_8.8550.pth',
#     'checkpoint_3999_8.9307.pth',
#     'checkpoint_3999_9.3629.pth',
#     'checkpoint_3999_9.7100.pth',
#     'checkpoint_3999_10.683.pth',
#     'checkpoint_3999_11.083.pth',
#
# ]



# PARA=[
#     'checkpoint_3800_6.6454.pth',
#     'checkpoint_3800_7.8179.pth',
#     'checkpoint_3800_7.8281.pth',
#     'checkpoint_3800_9.0281.pth',
#     'checkpoint_3800_9.4938.pth',
#     'checkpoint_3800_9.6853.pth',
#     'checkpoint_3800_10.882.pth',
#     'checkpoint_3800_10.901.pth',
#     'checkpoint_3800_11.091.pth',
#
#     'checkpoint_3999_6.8874.pth',
#     'checkpoint_3999_7.0259.pth',
#     'checkpoint_3999_7.6329.pth',
#     'checkpoint_3999_7.7672.pth',
#     'checkpoint_3999_7.8268.pth',
#     'checkpoint_3999_8.5023.pth',
#     'checkpoint_3999_8.8665.pth',
#     'checkpoint_3999_8.8767.pth',
#     'checkpoint_3999_9.9279.pth',
#
# ]

# PARA=[
#     # 'checkpoint_3800_6.5105_arc.pth',
#     # 'checkpoint_3800_7.5581_arc.pth',
#     # 'checkpoint_3800_8.4387_arc.pth',
#     # 'checkpoint_3800_9.0049_arc.pth',
#     # 'checkpoint_3800_9.6782_arc.pth',
#     # 'checkpoint_3800_9.9761_arc.pth',
#     # 'checkpoint_3800_10.735_arc.pth',
#     #
#     # 'checkpoint_3999_6.6454_arc.pth',
#     # 'checkpoint_3999_6.8906_arc.pth',
#     'checkpoint_3999_7.1074_arc.pth',
#     'checkpoint_3999_7.7115_arc.pth',
#     'checkpoint_3999_8.1664_arc.pth',
#     'checkpoint_3999_9.5297_arc.pth',
#     'checkpoint_3999_10.379_arc.pth',
#
# ]



# PARA=[
#     'checkpoint_3800_6.9332_cos.pth',
#     'checkpoint_3800_7.9332_cos.pth',
#     'checkpoint_3800_8.5845_cos.pth',
#     'checkpoint_3800_8.8934_cos.pth',
#     'checkpoint_3800_9.1841_cos.pth',
#     'checkpoint_3800_9.7818_cos.pth',
#     'checkpoint_3800_11.530_cos.pth',
#
#     'checkpoint_3999_7.1785_cos.pth',
#     'checkpoint_3999_8.1677_cos.pth',
#     'checkpoint_3999_8.5620_cos.pth',
#     'checkpoint_3999_8.6780_cos.pth',
#     'checkpoint_3999_9.0085_cos.pth',
#     'checkpoint_3999_9.1870_cos.pth',
#     'checkpoint_3999_9.3203_cos.pth',
#
# ]

#
# PARA=[
#     # 'checkpoint_3800_6.9332_cos.pth',
#     # 'checkpoint_3800_7.9332_cos.pth',
#     # 'checkpoint_3800_8.5845_cos.pth',
#     # 'checkpoint_3800_8.8934_cos.pth',
#     # 'checkpoint_3800_9.1841_cos.pth',
#     # 'checkpoint_3800_9.7818_cos.pth',
#     # 'checkpoint_3800_11.530_cos.pth',
#     #
#     # 'checkpoint_3999_7.1785_cos.pth',
#     # 'checkpoint_3999_8.1677_cos.pth',
#     # 'checkpoint_3999_8.5620_cos.pth',
#     # 'checkpoint_3999_8.6780_cos.pth',
#     # 'checkpoint_3999_9.0085_cos.pth',
#     # 'checkpoint_3999_9.1870_cos.pth',
#     'checkpoint_3999_10.862_cos.pth',
#
# ]

#
# PARA=[
#     # 'checkpoint_3800_6.9332_cos.pth',
#     # 'checkpoint_3800_7.9332_cos.pth',
#     # 'checkpoint_3800_8.5845_cos.pth',
#     # 'checkpoint_3800_8.8934_cos.pth',
#     # 'checkpoint_3800_9.1841_cos.pth',
#     # 'checkpoint_3800_9.7818_cos.pth',
#     # 'checkpoint_3800_11.530_cos.pth',
#     #
#     # 'checkpoint_3999_7.1785_cos.pth',
#     # 'checkpoint_3999_8.1677_cos.pth',
#     # 'checkpoint_3999_8.5620_cos.pth',
#     # 'checkpoint_3999_8.6780_cos.pth',
#     # 'checkpoint_3999_9.0085_cos.pth',
#     # 'checkpoint_3999_9.1870_cos.pth',
#     'checkpoint_3999_8.5695_arc.pth',
#
# ]

# PARA=[
#
#     'checkpoint_2999_0.4371.pth',
#     'checkpoint_3999_6.6100_arc.pth',
#     'checkpoint_3999_9.5943_cos.pth',
#
# ]

# PARA=[
#
#     'checkpoint_2999_0.3489.pth',
#     'checkpoint_3999_8.7359_arc.pth',
#     'checkpoint_3999_7.7515_cos.pth',
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
PARA=[
    'checkpoint_2800_0.5763.pth',
    'checkpoint_2999_0.6607.pth',

    'checkpoint_3800_9.0727_arc.pth',
    'checkpoint_3999_6.9830_arc.pth',

    'checkpoint_3800_9.2263_cos.pth',
    'checkpoint_3999_7.1840_cos.pth',

]

files_json=[
            # 'layer4_1_conv1_weight.json',
            'layer4_1_conv2_weight.json','layer4_2_conv2_weight.json',
            'layer4_conv2_weight.json',
            # 'layer4_1_conv3_weight.json',
            # 'layer4_2_conv3_weight.json',
            # 'layer4_2_conv1_weight.json',
            'embedding_max_avg.json',

            ]



results_csv=[
            # 'layer4_1_conv1_weight.csv',
            'layer4_1_conv2_weight.csv','layer4_2_conv2_weight.csv',
            'layer4_conv2_weight.csv',
            # 'layer4_1_conv3_weight.csv','layer4_2_conv3_weight.csv',
            # 'layer4_2_conv1_weight.csv',
            'embedding_max_avg.csv',
            ]







for index in range(len(PARA)):
    PARA_I = PARA[index].replace('.', '_')  # 避免路径出错
    target_path = join(os.path.abspath(os.path.dirname(os.getcwd())), 'datafile', PARA_I)

    for index_2 in range(len(files_json)):

        filename=join(target_path,files_json[index_2])
        print(filename)
        f = open(join(os.path.abspath(os.path.dirname(os.getcwd())),'result',results_csv[index_2]),'a+',encoding='utf-8',newline='' "")
        ff = open(join(os.path.abspath(os.path.dirname(os.getcwd())),'result',results_csv[index_2]),'r',encoding='utf-8',newline='' "")
        l=len(ff.readlines())
        ff.close()





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
        dataset_labels=test_labels

        #############################


        print(KDTree.valid_metrics)
        X = np.array(test_data)
        print(X.shape)



        ###########################################3
        # X1=np.array(train_data)
        # X=np.vstack((X,X1))
        # print(X.shape)
        #########################################




        query=np.array(test_data)
        # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        kdt = KDTree(X, leaf_size=40, metric='l2')
        # s = pickle.dumps(kdt)                     # doctest: +SKIP
        # kdt = pickle.loads(s)
        result=kdt.query(query, k=200, return_distance=False)
        #大概就是这样子的，有K个query,最终返回的a矩阵就有K行，每一行都有九个元素，代表
        #我们以第一行为例吧，第一行的九个元素。就代表距离第一个query距离最近的九张图片的索引（这九张图片来自训练集），我们通过这个索引
        #就可以获知这九张图片的具体的类别（如果我们已知了query的类别，既可以据此来计算mAP），
        # 以及对应的存储路径（在图像检索系统中进行最后的检索结果的输出）
        print(result[0:5])
        print(result.shape)
        query_result_labels=np.zeros(result.shape)
        h,w=result.shape
        for i in range(h):
            for j in range(w):
                query_result_labels[i,j]=dataset_labels[result[i,j]]
                # query_result_labels[i,j]=test_labels[result[i,j]]

        print(query_result_labels[0:5])


        # query_result_paths=[]
        # h,w=result.shape
        # for i in range(h):
        #     query_result_paths.append([])
        #     for j in range(w):
        #         query_result_paths[i].append(train_paths[result[i,j]])



        result_top_1=result[:,1:2]
        print(result_top_1.shape)
        result_top_1_labels=query_result_labels[:,1:2]
        print(result_top_1_labels.shape)

        ###################################################33
        result_top_2=result[:,1:3]
        print(result_top_2.shape)
        result_top_2_labels=query_result_labels[:,1:3]
        print(result_top_2_labels.shape)


        result_top_4=result[:,1:5]
        print(result_top_4.shape)
        result_top_4_labels=query_result_labels[:,1:5]
        print(result_top_4_labels.shape)


        result_top_8=result[:,1:9]
        print(result_top_8.shape)
        result_top_8_labels=query_result_labels[:,1:9]
        print(result_top_8_labels.shape)

        result_top_16=result[:,1:17]
        print(result_top_16.shape)
        result_top_16_labels=query_result_labels[:,1:17]
        print(result_top_16_labels.shape)

        result_top_32=result[:,1:33]
        print(result_top_32.shape)
        result_top_32_labels=query_result_labels[:,1:33]
        print(result_top_32_labels.shape)
        #####################################################
        # query_result_paths_top_1=[]
        # h,w=result.shape
        # for i in range(h):
        #     query_result_paths_top_1.append([])
        #     for j in range(1):
        #         query_result_paths_top_1[i].append(train_paths[result[i,j]])

        result_top_5=result[:,1:6]
        print(result_top_5.shape)
        result_top_5_labels=query_result_labels[:,1:6]
        print(result_top_5_labels[0:10,:])
        print(result_top_5_labels.shape)
        # query_result_paths_top_5=[]
        # h,w=result.shape
        # for i in range(h):
        #     query_result_paths_top_5.append([])
        #     for j in range(5):
        #         query_result_paths_top_5[i].append(train_paths[result[i,j]])
        result_top_10=result[:,1:11]
        result_top_10_labels=query_result_labels[:,1:11]

        result_top_15=result[:,1:16]
        result_top_15_labels=query_result_labels[:,1:16]

        result_top_20=result[:,1:21]
        result_top_20_labels=query_result_labels[:,1:21]


        result_top_30=result[:,1:31]
        result_top_30_labels=query_result_labels[:,1:31]

        result_top_40=result[:,1:41]
        result_top_40_labels=query_result_labels[:,1:41]

        result_top_50=result[:,1:51]
        result_top_50_labels=query_result_labels[:,1:51]

        result_top_60=result[:,1:61]
        result_top_60_labels=query_result_labels[:,1:61]

        result_top_70=result[:,1:71]
        result_top_70_labels=query_result_labels[:,1:71]

        result_top_80=result[:,1:81]
        result_top_80_labels=query_result_labels[:,1:81]

        result_top_90=result[:,1:91]
        result_top_90_labels=query_result_labels[:,1:91]

        result_top_100=result[:,1:101]
        result_top_100_labels=query_result_labels[:,1:101]

        result_top_200=result[:,1:201]
        result_top_200_labels=query_result_labels[:,1:201]

        # result_top_500=result[:,1:501]
        # result_top_500_labels=query_result_labels[:,1:501]

        def compute_mAP(querys_results,querys_labels,querys_results_labels):
            '''query_results:[5794,9]的array
                query_results_labels:[5794,9]的array
                query_labels:[5794,]的list
                   pos_list:数据库中与查询图像相似的结果'''
            querys_labels=np.array(querys_labels)
            one_precision = []
            intersect_size = 0
            precision_all=[]
            for i in range(querys_results.shape[0]):#5794
                intersect_size = 0
                one_precision = []
                for j in range(querys_results.shape[1]):#9
                    if querys_results_labels[i,j] == querys_labels[i]:
                        intersect_size =intersect_size + 1
                        # precision = intersect_size - 1 / (j)
                        precision = intersect_size / (j + 1 )

                        one_precision.append(precision)
                if len(one_precision)==0:
                    precision_all.append(0)
                else:
                    # precision_all.append(np.mean(np.array(one_precision)).tolist())
                    precision_all.append(np.mean(np.array(one_precision)))

            return np.mean(np.array(precision_all))


        #Recall@K
        def compute_mAP_new(querys_results,querys_labels,querys_results_labels):
            '''query_results:[5794,9]的array
                query_results_labels:[5794,9]的array
                query_labels:[5794,]的list
                   pos_list:数据库中与查询图像相似的结果'''
            querys_labels=np.array(querys_labels)
            score=0
            for i in range(querys_results.shape[0]):#5794
                if querys_labels[i].tolist() in  querys_results_labels[i,:].tolist():
                    score=score+1

            return  score/querys_results.shape[0]


        print("top_1 mAP:")
        top_1=compute_mAP(result_top_1,test_labels,result_top_1_labels)
        print(top_1)

        print("top_5 mAP:")
        top_5=compute_mAP(result_top_5,test_labels,result_top_5_labels)
        print(top_5)

        print("top_10 mAP:")
        top_10=compute_mAP(result_top_10,test_labels,result_top_10_labels)
        print(top_10)

        print("top_15 mAP:")
        top_15=compute_mAP(result_top_15,test_labels,result_top_15_labels)
        print(top_15)

        print("top_20 mAP:")
        top_20=compute_mAP(result_top_20,test_labels,result_top_20_labels)
        print(top_20)

        print("top_30 mAP:")
        top_30=compute_mAP(result_top_30,test_labels,result_top_30_labels)
        print(top_30)

        print("top_40 mAP:")
        top_40=compute_mAP(result_top_40,test_labels,result_top_40_labels)
        print(top_40)

        print("top_50 mAP:")
        top_50=compute_mAP(result_top_50,test_labels,result_top_50_labels)
        print(top_50)

        print("top_60 mAP:")
        top_60=compute_mAP(result_top_60,test_labels,result_top_60_labels)
        print(top_60)

        print("top_70 mAP:")
        top_70=compute_mAP(result_top_70,test_labels,result_top_70_labels)
        print(top_70)

        print("top_80 mAP:")
        top_80=compute_mAP(result_top_80,test_labels,result_top_80_labels)
        print(top_80)

        print("top_90 mAP:")
        top_90=compute_mAP(result_top_90,test_labels,result_top_90_labels)
        print(top_90)

        print("top_100 mAP:")
        top_100=compute_mAP(result_top_100,test_labels,result_top_100_labels)
        print(top_100)

        print("top_200 mAP:")
        top_200=compute_mAP(result_top_200,test_labels,result_top_200_labels)
        print(top_200)

        # print("top_500 mAP:")
        # print(compute_mAP(result_top_500,test_labels,result_top_500_labels))
        print("*-*-*-*-*-*--*-*-*-*-*--*-*--*--*--*-*--*--*--*--*--*-*-*---*-*-*-")
        # print("top_1 mAP_new:")
        # print(compute_mAP_new(result_top_1,test_labels,result_top_1_labels))
        # print("top_5 mAP_new:")
        # print(compute_mAP_new(result_top_5,test_labels,result_top_5_labels))
        print("Recall@1:")
        R_1=compute_mAP_new(result_top_1,test_labels,result_top_1_labels)
        print(R_1)
        print("Recall@2:")
        R_2=compute_mAP_new(result_top_2,test_labels,result_top_2_labels)
        print(R_2)
        print("Recall@4:")
        R_4=compute_mAP_new(result_top_4,test_labels,result_top_4_labels)
        print(R_4)
        print("Recall@8:")
        R_8=compute_mAP_new(result_top_8,test_labels,result_top_8_labels)
        print(R_8)
        print("Recall@16:")
        R_16=compute_mAP_new(result_top_16,test_labels,result_top_16_labels)
        print(R_16)
        print("Recall@32:")
        R_32=compute_mAP_new(result_top_32,test_labels,result_top_32_labels)
        print(R_32)

        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        csv_writer = csv.writer(f)
        if l==0:
            csv_writer.writerow(["top_1","top_5","top_10","top_15","top_20","top_30","top_40","top_50","top_60","top_70","top_80","top_90","top_100","top_200",'R@1','R@2','R@4','R@8','R@16','R@32'])
        csv_writer.writerow([top_1,top_5,top_10,top_15,top_20,top_30,top_40,top_50,top_60,top_70,top_80,top_90,top_100,top_200,R_1,R_2,R_4,R_8,R_16,R_32])

        f.close()

