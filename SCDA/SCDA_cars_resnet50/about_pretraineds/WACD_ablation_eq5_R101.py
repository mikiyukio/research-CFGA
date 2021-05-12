from SCDA_cars_resnet50.about_pretraineds.net_res50 import FGIAnet100_metric5_1, \
    FGIAnet100_metric5, FGIAnet100_metric5_2, FGIAnet100_R18, FGIAnet100_R34, FGIAnet100_R101
from SCDA_cars_resnet50 import dataloader
from SCDA_cars_resnet50 import dataloader_test  # DGCRL test  无监督test
from SCDA_cars_resnet50 import dataloader_DGPCRL_test  # DGPCRL test

from SCDA_cars_resnet50.bwconncomp import largestConnectComponent
import argparse
from os.path import join
import uuid
import time
import json
import pickle
import sys, os

sys.path.append(os.pardir)
import torch
from torch.nn.functional import interpolate
import torch.nn as nn
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
import cv2
from SCDA_cars_resnet50.about_pretraineds.grad_cam import PoolCam_yuhan_kernel_scda_origin, \
    PoolCam_yuhan_kernel_scda_origin_no_lcc, \
    PoolCam_yuhan_kernel_scda_origin_R18, \
    PoolCam_yuhan_kernel_scda_origin_R34, \
    PoolCam_yuhan_kernel_scda_origin_R101_eq5

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=700, help='进行特征提取的图片的尺寸的上界所对应的数量级')
# parser.add_argument('--img_size', type=int, default = 280,help='进行特征提取的图片的尺寸的上界所对应的数量级')


PARA = [
    'resnet101-5d3b4d8f.pth'
]

parser.add_argument('--savedir', default='./models/', help="Path to save weigths and logs")
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.",
                    default=PARA)
args = parser.parse_args()

# args = parser.parse_args()


net = FGIAnet100_R101()
# net = FGIAnet100_metric5_2(scale=100)
# net = FGIAnet100_metric5_2()

for index in range(len(PARA)):

    # #SCDA_pretrained_model
    save_model = torch.load(join(os.path.abspath(os.path.dirname(os.getcwd())), args.savedir,  'resnet101-5d3b4d8f.pth'))
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    model_dict.update(state_dict)  # 这样就对我们自定义网络的cnn部分的参数进行了更新，更新为vgg16网络中cnn部分的参数值
    net.load_state_dict(model_dict)

    ####################################################################################
    net.eval()
    net.cuda()

    train_paths_name = join(os.path.abspath(os.path.dirname(os.getcwd())), './datafile/train_paths.json')
    test_paths_name = join(os.path.abspath(os.path.dirname(os.getcwd())), './datafile/test_paths.json')
    train_labels_name = join(os.path.abspath(os.path.dirname(os.getcwd())), './datafile/train_labels.json')
    test_labels_name = join(os.path.abspath(os.path.dirname(os.getcwd())), './datafile/test_labels.json')
    # train_paths_name='./datafile/train_paths.json'
    # test_paths_name='./datafile/test_paths.json'
    # train_labels_name='./datafile/train_labels.json'
    # test_labels_name='./datafile/test_labels.json'
    with open(train_paths_name) as miki:
        train_paths = json.load(miki)
    with open(test_paths_name) as miki:
        test_paths = json.load(miki)
    with open(train_labels_name) as miki:
        train_labels = json.load(miki)
    with open(test_labels_name) as miki:
        test_labels = json.load(miki)
    # loaders = dataloader.get_dataloaders(train_paths, test_paths,train_labels,test_labels,args.img_size, 1,1,1,SCDA_flag=1)#返回值为由可迭代DataLoader对象所组成的字典
    # loaders = dataloader_test.get_dataloaders(train_paths, test_paths,train_labels,test_labels,args.img_size, 1,1,1,SCDA_flag=1)#返回值为由可迭代DataLoader对象所组成的字典
    loaders = dataloader_test.get_dataloaders(train_paths, test_paths, train_labels, test_labels, args.img_size, 1, 1,
                                              1, SCDA_flag=1)  # 返回值为由可迭代DataLoader对象所组成的字典

    net_dict = net.state_dict()
    for key, value in net_dict.items():
        print(key, '\t', net.state_dict()[key].size())

    print("cnn model is ready.")
    num_tr = len(loaders['train'].dataset)  # %num_tr便是训练集所对应的图片的个数
    num_te = len(loaders['test'].dataset)  # %num_te便是测试集所对应的图片的个数

    tr_L28_mean = []
    te_L28_mean = []
    tr_L31_mean = []
    te_L31_mean = []

    tr4_1_1 = []
    tr4_1_3 = []
    tr4_2_1 = []
    tr4_2_3 = []
    te4_1_1 = []
    te4_1_3 = []
    te4_2_1 = []
    te4_2_3 = []

    tr_L28_mean_2 = []
    te_L28_mean_2 = []
    tr_L31_mean_2 = []
    te_L31_mean_2 = []

    tr_L31_mean_3 = []
    te_L31_mean_3 = []

    tr_L31_mean_3_1 = []
    te_L31_mean_3_1 = []

    ii = 0
    for phase in ['test']:
        # for phase in ['train','test']:
        for images, labels in loaders[phase]:  # 一个迭代器，迭代数据集中的每一batch_size图片;迭代的返回值dataset的子类中的__getitem__()方法是如何进行重写的；
            print(ii)
            ii = ii + 1
            # print(images.size())
            for flip in range(1):
                if flip == 0:
                    pass
                else:
                    images = images[:, :, :, torch.arange(images.size(3) - 1, -1, -1).long()]  # 整个batch_size的所有图像水平翻转
                # print(images[0].size())
                # image=images[0]#去除了batch_size那一维度，反正batch_size都是1，有没有无所谓
                batch_size, c, h, w = images.size()
                if min(h, w) > args.img_size:
                    images = interpolate(images, size=[int(h * (args.img_size / min(h, w))),
                                                       int(w * (args.img_size / min(h, w)))], mode="bilinear",
                                         align_corners=True)
                    # %我就打个比方吧，min(h,w)=h的话  h*(700/min(h,w)=700   w*(700/min(h,w)=w*(700/h)=700*(w/h) 图像的size由[h,w]变为[700,700*(w/h)]
                    # %由此可见，在min(h,w) > 700的前提下，图像被适当的进行分辨率的缩小，到700这一级，但是长宽比是没有改变的，图像没有变形
                    # %这一步操作只是为了对于图像的分辨率的上限进行一个限制
                batch_size, c, h, w = images.size()
                # matlab版本的实现中这里是对可能出现的灰度图像进行通道数扩充，并减去图像在各个通道上的均值，以上过程我在dataloader.py中已经实现了，以上。
                images, labels = images.cuda(), labels.cuda()
                labels = torch.zeros_like(labels).cuda()
                # labels=labels.cpu()

                grad_cam_scda = PoolCam_yuhan_kernel_scda_origin_R101_eq5(net)  # 有最大连通域
                # grad_cam_scda = PoolCam_yuhan_kernel_scda_origin_no_lcc(net)# 无最大连通域
                # grad_cam_scda = GradCam_yuhan_kernel_version30_DGCRL_scda_many_5(net)#####论文用的

                ##########################

                # cam_L28, cam_L31, feature_maps_L28, feature_maps_L31=grad_cam_scda(images)
                output_mean, output_mean_2, output_mean_3 = grad_cam_scda(images, ii=ii, flip=flip)
                # output_mean_26 与output_mean_28 分别存储着来自,'layer4.2.conv2.weight','layer4.1.conv2.weight'的卷积核梯度特征
                f4_1_1, output_mean_28, f4_1_3, f4_2_1, output_mean_26, f4_2_3 = output_mean[0], output_mean[1], \
                                                                                 output_mean[2], output_mean[3], \
                                                                                 output_mean[4], output_mean[5]
                # output_mean_26_2 , output_mean_28_2是resnet50的backzone输出的特征图所得到的scda特征，
                # 分别是
                # #先norm再拼接
                # 与
                # #先拼接再norm
                # 所对应的特征
                output_mean_26_2, output_mean_28_2 = output_mean_2[0], output_mean_2[1]
                embedding_mean_max, embedding_mean_max_1 = output_mean_3[0], output_mean_3[1]

                feature_maps_L28_mean_norm, feature_maps_L31_mean_norm = output_mean_26, output_mean_28
                feature_maps_L28_2_mean_norm, feature_maps_L31_2_mean_norm = output_mean_26_2, output_mean_28_2

                if phase == 'train':
                    if flip == 0:
                        tr_L31_mean.append(feature_maps_L31_mean_norm.tolist())
                        tr_L28_mean.append(feature_maps_L28_mean_norm.tolist())
                        tr_L31_mean_2.append(feature_maps_L31_2_mean_norm.tolist())
                        tr_L28_mean_2.append(feature_maps_L28_2_mean_norm.tolist())
                        tr4_1_1.append(f4_1_1.tolist())
                        tr4_1_3.append(f4_1_3.tolist())
                        tr4_2_1.append(f4_2_1.tolist())
                        tr4_2_3.append(f4_2_3.tolist())
                        tr_L31_mean_3.append(embedding_mean_max.tolist())
                        tr_L31_mean_3_1.append(embedding_mean_max_1.tolist())

                    else:
                        pass
                else:
                    if flip == 0:
                        te_L31_mean.append(feature_maps_L31_mean_norm.tolist())
                        te_L28_mean.append(feature_maps_L28_mean_norm.tolist())
                        te_L31_mean_2.append(feature_maps_L31_2_mean_norm.tolist())
                        te_L28_mean_2.append(feature_maps_L28_2_mean_norm.tolist())
                        te4_1_1.append(f4_1_1.tolist())
                        te4_1_3.append(f4_1_3.tolist())
                        te4_2_1.append(f4_2_1.tolist())
                        te4_2_3.append(f4_2_3.tolist())
                        te_L31_mean_3.append(embedding_mean_max.tolist())
                        te_L31_mean_3_1.append(embedding_mean_max_1.tolist())

                    else:
                        pass
    print("save starting..........................................")

    print('SCDA avgPool and maxpool for trainset and dataset is done................................')
    print('stacking starting...............................................')

    tr_L31_mean = np.array(tr_L31_mean)
    te_L31_mean = np.array(te_L31_mean)
    tr_L28_mean = np.array(tr_L28_mean)
    te_L28_mean = np.array(te_L28_mean)

    tr_L31_mean_2 = np.array(tr_L31_mean_2)
    te_L31_mean_2 = np.array(te_L31_mean_2)
    tr_L28_mean_2 = np.array(tr_L28_mean_2)
    te_L28_mean_2 = np.array(te_L28_mean_2)

    tr_L31_mean_3 = np.array(tr_L31_mean_3)
    te_L31_mean_3 = np.array(te_L31_mean_3)

    tr_L31_mean_3_1 = np.array(tr_L31_mean_3_1)
    te_L31_mean_3_1 = np.array(te_L31_mean_3_1)

    tr4_1_1 = np.array(tr4_1_1)
    tr4_1_3 = np.array(tr4_1_3)
    tr4_2_1 = np.array(tr4_2_1)
    tr4_2_3 = np.array(tr4_2_3)
    te4_1_1 = np.array(te4_1_1)
    te4_1_3 = np.array(te4_1_3)
    te4_2_1 = np.array(te4_2_1)
    te4_2_3 = np.array(te4_2_3)

    PARA_I = PARA[index].replace('.', '_')  # 避免路径出错
    # target_path = os.path.join(target, str(clas) + class_name)
    target_path = join(os.path.abspath(os.path.dirname(os.getcwd())), 'datafile', PARA_I)

    if not os.path.isdir(target_path):
        os.mkdir(target_path)
    print(target_path)

    train_data = tr4_1_1.tolist()
    print('train_data.shape:', np.array(train_data).shape)
    test_data = te4_1_1.tolist()
    print('test_data.shape:', np.array(test_data).shape)
    final_features = {}
    final_features['train'] = train_data
    final_features['test'] = test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_1_conv1_weight.json')#'layer4.1.conv1.weight'
    filename = join(target_path, 'layer4_1_conv1_weight.json')  # 'layer4.1.conv1.weight'

    with open(filename, 'w') as f_obj:
        json.dump(final_features, f_obj)

    train_data = tr4_1_3.tolist()
    print('train_data.shape:', np.array(train_data).shape)
    test_data = te4_1_3.tolist()
    print('test_data.shape:', np.array(test_data).shape)
    final_features = {}
    final_features['train'] = train_data
    final_features['test'] = test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_1_conv3_weight.json')#'layer4.1.conv3.weight'
    filename = join(target_path, 'layer4_1_conv3_weight.json')  # 'layer4.1.conv3.weight'

    with open(filename, 'w') as f_obj:
        json.dump(final_features, f_obj)

    train_data = tr4_2_3.tolist()
    print('train_data.shape:', np.array(train_data).shape)
    test_data = te4_2_3.tolist()
    print('test_data.shape:', np.array(test_data).shape)
    final_features = {}
    final_features['train'] = train_data
    final_features['test'] = test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_2_conv3_weight.json')#'layer4.2.conv3.weight'
    filename = join(target_path, 'layer4_2_conv3_weight.json')  # 'layer4.1.conv3.weight'

    with open(filename, 'w') as f_obj:
        json.dump(final_features, f_obj)

    train_data = tr4_2_1.tolist()
    print('train_data.shape:', np.array(train_data).shape)
    test_data = te4_2_1.tolist()
    print('test_data.shape:', np.array(test_data).shape)
    final_features = {}
    final_features['train'] = train_data
    final_features['test'] = test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_2_conv1_weight.json')#'layer4.2.conv1.weight'
    filename = join(target_path, 'layer4_2_conv1_weight.json')  # 'layer4.1.conv3.weight'

    with open(filename, 'w') as f_obj:
        json.dump(final_features, f_obj)

    train_data = tr_L31_mean.tolist()
    print('train_data.shape:', np.array(train_data).shape)
    test_data = te_L31_mean.tolist()
    print('test_data.shape:', np.array(test_data).shape)
    final_features = {}
    final_features['train'] = train_data
    final_features['test'] = test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_1_conv2_weight.json')#'layer4.1.conv2.weight'
    filename = join(target_path, 'layer4_1_conv2_weight.json')  # 'layer4.1.conv3.weight'

    with open(filename, 'w') as f_obj:
        json.dump(final_features, f_obj)

    train_data = tr_L28_mean.tolist()
    print('train_data.shape:', np.array(train_data).shape)
    test_data = te_L28_mean.tolist()
    print('test_data.shape:', np.array(test_data).shape)
    final_features = {}
    final_features['train'] = train_data
    final_features['test'] = test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_2_conv2_weight.json')#'layer4.2.conv2.weight'
    filename = join(target_path, 'layer4_2_conv2_weight.json')  # 'layer4.1.conv3.weight'

    with open(filename, 'w') as f_obj:
        json.dump(final_features, f_obj)

    train_data = np.hstack([tr_L31_mean,
                            tr_L28_mean,
                            ]).tolist()
    print('train_data.shape:', np.array(train_data).shape)
    test_data = np.hstack([te_L31_mean,
                           te_L28_mean,
                           ]).tolist()
    print('test_data.shape:', np.array(test_data).shape)

    final_features = {}
    final_features['train'] = train_data
    final_features['test'] = test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_conv2_weight.json')##'layer4.1.conv2.weight' + #'layer4.2.conv2.weight'
    filename = join(target_path, 'layer4_conv2_weight.json')  # 'layer4.1.conv3.weight'

    with open(filename, 'w') as f_obj:
        json.dump(final_features, f_obj)

    train_data = tr_L31_mean_2.tolist()
    print('train_data.shape:', np.array(train_data).shape)
    test_data = te_L31_mean_2.tolist()
    print('test_data.shape:', np.array(test_data).shape)

    final_features = {}
    final_features['train'] = train_data
    final_features['test'] = test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')
    filename = join(target_path, 'scda_max_avg_norm.json')  # 'layer4.1.conv3.weight'

    with open(filename, 'w') as f_obj:
        json.dump(final_features, f_obj)

    train_data = tr_L28_mean_2.tolist()
    print('train_data.shape:', np.array(train_data).shape)
    test_data = te_L28_mean_2.tolist()
    print('test_data.shape:', np.array(test_data).shape)

    final_features = {}
    final_features['train'] = train_data
    final_features['test'] = test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')
    filename = join(target_path, 'scda_norm_max_avg.json')  # 'layer4.1.conv3.weight'

    with open(filename, 'w') as f_obj:
        json.dump(final_features, f_obj)

    train_data = tr_L31_mean_3.tolist()
    print('train_data.shape:', np.array(train_data).shape)
    test_data = te_L31_mean_3.tolist()
    print('test_data.shape:', np.array(test_data).shape)

    final_features = {}
    final_features['train'] = train_data
    final_features['test'] = test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')
    filename = join(target_path, 'embedding_max_avg.json')  # 'layer4.1.conv3.weight'

    with open(filename, 'w') as f_obj:
        json.dump(final_features, f_obj)

    train_data = tr_L31_mean_3_1.tolist()
    print('train_data.shape:', np.array(train_data).shape)
    test_data = te_L31_mean_3_1.tolist()
    print('test_data.shape:', np.array(test_data).shape)

    final_features = {}
    final_features['train'] = train_data
    final_features['test'] = test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')
    filename = join(target_path, 'embedding_norm_max_avg.json')  # 'layer4.1.conv3.weight'

    with open(filename, 'w') as f_obj:
        json.dump(final_features, f_obj)




