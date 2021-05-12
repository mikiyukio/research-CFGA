import numpy as np
from sklearn.neighbors import KDTree
import json
import matplotlib.pyplot as plt
# from SCDA_cars_resnet50.net_res50 import FGIAnet100_metric5_1,FGIAnet100_metric5,FGIAnet100_metric5_2
from SCDA_cars_resnet50.about_msloss.net_res50 import FGIAnet100_metric5_ms
from SCDA_cars_resnet50.about_msloss import dataloader_msloss
from SCDA_cars_resnet50.bwconncomp import largestConnectComponent
import os
import argparse
from os.path import join
import uuid
import json
# from SCDA_cars_resnet50 import dataloader_DGPCRL_test
from torch.nn.functional import interpolate
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from SCDA_cars_resnet50.grad_cam import GradCam_yuhan_kernel_version30_DGCRL_scda_many_for_retrivial,\
    GradCam_pp,GradCam,GradCam_new,GradCam_yuhan_kernel_version30_amsoftmax_scda,\
    GradCam_yuhan_kernel_version30_amsoftmax_scda_for_retrivial,\
    msloss_for_retrieval,PoolCam_yuhan_kernel_for_retrieval
import cv2

parser = argparse.ArgumentParser()
# parser = argparse.ArgumentParser()
parser.add_argument('--datasetdir', default=r'C:\Users\于涵\Desktop\Stanford car dataset\car_ims',  help="path to cub200_2011 dir")
parser.add_argument('--imgdir', default='cars_196',  help="path to train img dir")
parser.add_argument('--tr_te_split_txt', default=r'train_test_split.txt',  help="关于训练集与测试集的划分，0代表测试集，1代表训练集")
parser.add_argument('--tr_te_image_name_txt', default=r'images.txt',  help="关于训练集与测试集的图片的相对路径名字")
parser.add_argument('--image_labels_txt', default=r'image_class_labels.txt',  help="图像的类别标签标记")
parser.add_argument('--class_name_txt', default=r'classes.txt',  help="图像的200个类别名称")
parser.add_argument("--num_classes", type=int, dest="num_classes", help="Total number of epochs to train", default=200)
parser.add_argument('--img_size', type=int, default = 700,help='进行特征提取的图片的尺寸的上界所对应的数量级')
# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_11_0.6863.pth')
# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_15_1.0.pth')
# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_120_0.9943.pth')
# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_112_0.9433.pth')
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_2999_0.4325.pth')
# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_3800_9.1800_cos.pth')

parser.add_argument('--savedir',default='./models/', help="Path to save weigths and logs")
args = parser.parse_args()

# net = FGIAnet100_metric5()
# net = FGIAnet100_metric5_1()
# net = FGIAnet100_metric5_2(scale=100)
net = FGIAnet100_metric5_ms()

# net = FGIAnet100_metric5_2(scale=128)

#SCDA_CAM
checkpoint = torch.load(join(os.path.abspath(os.path.dirname(os.getcwd())),args.savedir,args.model_name))
net.load_state_dict(checkpoint['model'])


#
# save_model = torch.load(join(os.path.abspath(os.path.dirname(os.getcwd())),args.savedir,'vgg16-397923af.pth'))
# model_dict =  net.state_dict()
# state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
# print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
# model_dict.update(state_dict)#这样就对我们自定义网络的cnn部分的参数进行了更新，更新为vgg16网络中cnn部分的参数值
# net.load_state_dict(model_dict)



net.eval()
net.cuda()
# img_name=''
# image = Image.open(img_name).convert('RGB')
# print(image.shape)
# totensor=transforms.ToTensor()
# image=totensor(image)
# batch_size, c, h, w = image.size()
# image = image.cuda()
# _, cnnFeature_maps = net(image)
# feature_maps_L28=cnnFeature_maps[0] #如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
# batch_size, c_L28, h_L28, w_L28=feature_maps_L28.size()
# feature_maps_L31=cnnFeature_maps[1]##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
# batch_size, c_L31, h_L31, w_L31=feature_maps_L31.size()


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





#上面的这些行都没用，主要是指定test_paths,这里面指定这我们所要检索的query所在的路径
# img_name=r'100FIAT_500_Convertible_2012\008120.jpg'
# img_name =r'6Acura_Integra_Type_R_2001\000428.jpg'
# img_name = r'24Audi_S4_Sedan_2007\001892.jpg'
# img_name=r'115Ford_Focus_Sedan_2007\009423.jpg'
# img_name=r'115Ford_Focus_Sedan_2007\009427.jpg'
# img_name=r'115Ford_Focus_Sedan_2007\009377.jpg'
# img_name = r'115Ford_Focus_Sedan_2007\009452.jpg'
# img_name = r'115Ford_Focus_Sedan_2007\009455.jpg'
# img_name = r'115Ford_Focus_Sedan_2007\009384.jpg'
# img_name = r'115Ford_Focus_Sedan_2007\009448.jpg'
# img_name = r'115Ford_Focus_Sedan_2007\009421.jpg'
img_name = r'115Ford_Focus_Sedan_2007\009390.jpg'
# img_name = r'115Ford_Focus_Sedan_2007\009396.jpg'
# img_name =r'111Ford_Ranger_SuperCab_2011\009042.jpg'
# img_name =r'111Ford_Ranger_SuperCab_2011\009084.jpg'
# img_name =r'111Ford_Ranger_SuperCab_2011\009088.jpg'
# img_name  = r'142Infiniti_QX56_SUV_2011\011636.jpg'
# img_name = r'100FIAT_500_Convertible_2012\008135.jpg'
# img_name = r'100FIAT_500_Convertible_2012\008134.jpg'
# img_name = r'100FIAT_500_Convertible_2012\008146.jpg'
# img_name = r'180Spyker_C8_Coupe_2009\014834.jpg'
# img_name= r'160McLaren_MP4-12C_Coupe_2012\013080.jpg'
# img_name=r'16Audi_V8_Sedan_1994\001290.jpg'
# img_name=r'16Audi_V8_Sedan_1994\001233.jpg'
# img_name = r'39Bentley_Continental_Supersports_Conv._Convertible_2012\\003112.jpg'
# img_name = r'39Bentley_Continental_Supersports_Conv._Convertible_2012\003118.jpg'
# img_name = r'45Bugatti_Veyron_16.4_Convertible_2009\003635.jpg'
# img_name = r'60Chevrolet_HHR_SS_2010\004867.jpg'

# img_name = r'15Audi_R8_Coupe_2012\001192.jpg'
# img_name = r'15Audi_R8_Coupe_2012\001184.jpg'
# img_name = r'15Audi_R8_Coupe_2012\001156.jpg'
# img_name = r'15Audi_R8_Coupe_2012\001160.jpg'
# img_name = r'15Audi_R8_Coupe_2012\001170.jpg'
# img_name = r'15Audi_R8_Coupe_2012\001172.jpg'
# img_name = r'15Audi_R8_Coupe_2012\001194.jpg'
# img_name = r'15Audi_R8_Coupe_2012\001141.jpg'
# img_name = r'15Audi_R8_Coupe_2012\001198.jpg'
# img_name = r'15Audi_R8_Coupe_2012\001208.jpg'
# img_name = r'15Audi_R8_Coupe_2012\001211.jpg'

# img_name=r'19Audi_TT_Hatchback_2011\001518.jpg'
# img_name=r'85Dodge_Caravan_Minivan_1997\006894.jpg'
# img_name=r'85Dodge_Caravan_Minivan_1997\006877.jpg'
# img_name=r'85Dodge_Caravan_Minivan_1997\006882.jpg'
# img_name=r'85Dodge_Caravan_Minivan_1997\006899.jpg'
# img_name=r'85Dodge_Caravan_Minivan_1997\006938.jpg'
# img_name=r'85Dodge_Caravan_Minivan_1997\006941.jpg'
# img_name=r'85Dodge_Caravan_Minivan_1997\006879.jpg'
# img_name=r'85Dodge_Caravan_Minivan_1997\006909.jpg'
# img_name=r'85Dodge_Caravan_Minivan_1997\006908.jpg'
# img_name=r'85Dodge_Caravan_Minivan_1997\006913.jpg'
# img_name=r'85Dodge_Caravan_Minivan_1997\006893.jpg'
# img_name=r'85Dodge_Caravan_Minivan_1997\006949.jpg'
# img_name=r'85Dodge_Caravan_Minivan_1997\006937.jpg'
# img_name=r'85Dodge_Caravan_Minivan_1997\006936.jpg'

# img_name=r'88Dodge_Sprinter_Cargo_Van_2009\007203.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007396.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007398.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007442.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007443.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007453.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007454.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007456.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007460.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007439.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007390.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007420.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007421.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007411.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007416.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007418.jpg'
# img_name=r'91Dodge_Dakota_Club_Cab_2007\007419.jpg'

# img_name = r'39Bentley_Continental_Supersports_Conv._Convertible_2012\003118.jpg'
# img_name = r'39Bentley_Continental_Supersports_Conv._Convertible_2012\003112.jpg'
# img_name = r'39Bentley_Continental_Supersports_Conv._Convertible_2012\003120.jpg'
# img_name = r'39Bentley_Continental_Supersports_Conv._Convertible_2012\003123.jpg'
# img_name = r'39Bentley_Continental_Supersports_Conv._Convertible_2012\003133.jpg'
# img_name = r'39Bentley_Continental_Supersports_Conv._Convertible_2012\003171.jpg'
# img_name = r'39Bentley_Continental_Supersports_Conv._Convertible_2012\003178.jpg'
# img_name = r'39Bentley_Continental_Supersports_Conv._Convertible_2012\003160.jpg'
# img_name = r'39Bentley_Continental_Supersports_Conv._Convertible_2012\003172.jpg'
# img_name = r'39Bentley_Continental_Supersports_Conv._Convertible_2012\003145.jpg'
# img_name = r'39Bentley_Continental_Supersports_Conv._Convertible_2012\003149.jpg'

# img_name = r'1AM_General_Hummer_SUV_2000\000009.jpg'
# img_name = r'1AM_General_Hummer_SUV_2000\000083.jpg'
# img_name = r'1AM_General_Hummer_SUV_2000\000004.jpg'
# img_name = r'1AM_General_Hummer_SUV_2000\000021.jpg'
# img_name = r'1AM_General_Hummer_SUV_2000\000070.jpg'

# img_name=r'185Tesla_Model_S_Sedan_2012\015175.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015226.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015218.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015229.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015209.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015243.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015246.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015199.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015207.jpg'


# img_name=r'168Nissan_Leaf_Hatchback_2012\013811.jpg'

# img_name=r'D:\python_work\SCDA\SCDA_pro\retrivial_visualize\croped_images\422_1.jpg'

# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014496.jpg'
# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014506.jpg'
# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014508.jpg'
# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014502.jpg'
# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014519.jpg'
# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014551.jpg'
# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014575.jpg'
# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014510.jpg'

# img_name=r'144Jaguar_XK_XKR_2012\011847.jpg'
# img_name=r'159Mazda_Tribute_SUV_2011\013055.jpg'
# img_name = r'149Jeep_Compass_SUV_2012\012206.jpg'

# img_name=r'123Geo_Metro_Convertible_1993\010128.jpg'
# img_name=r'123Geo_Metro_Convertible_1993\010120.jpg'
# img_name=r'123Geo_Metro_Convertible_1993\010129.jpg'
# img_name=r'123Geo_Metro_Convertible_1993\010182.jpg'
# img_name=r'192Volkswagen_Beetle_Hatchback_2012/015841.jpg'
# img_name=r'192Volkswagen_Beetle_Hatchback_2012/015843.jpg'
# img_name=r'192Volkswagen_Beetle_Hatchback_2012/015762.jpg'
# img_name=r'157MINI_Cooper_Roadster_Convertible_2012/012906.jpg'
# img_name=r'160McLaren_MP4-12C_Coupe_2012/013075.jpg'
# img_name=r'160McLaren_MP4-12C_Coupe_2012/013118.jpg'










test_path=[]
test_path.append(join(args.datasetdir,args.imgdir,img_name))
# test_path.append(img_name)

loaders = dataloader_msloss.get_dataloaders(train_paths, test_path,train_labels,test_labels,args.img_size, 1,1,1,SCDA_flag=1)#返回值为由可迭代DataLoader对象所组成的字典



print("cnn model is ready.")

tr_L28_mean = []
te_L28_mean = []
tr_L31_mean = []
te_L31_mean = []

tr4_1_1=[]
tr4_1_3=[]
tr4_2_1=[]
tr4_2_3=[]
te4_1_1=[]
te4_1_3=[]
te4_2_1=[]
te4_2_3=[]


tr_L28_mean_2 = []
te_L28_mean_2 = []
tr_L31_mean_2 = []
te_L31_mean_2 = []

tr_L31_mean_3 = []
te_L31_mean_3 = []

tr_L31_mean_3_1 = []
te_L31_mean_3_1 = []
yuri=2040


ii=0
for phase in ['test']:
# for phase in ['train','test']:
        for images, labels in loaders[phase]:#一个迭代器，迭代数据集中的每一batch_size图片;迭代的返回值dataset的子类中的__getitem__()方法是如何进行重写的；
                print(ii)
                ii=ii+1
                for flip in range(1):
                        if flip==0:
                                pass
                        else:
                                images=images[:,:,:,torch.arange(images.size(3)-1,-1,-1).long()]  #整个batch_size的所有图像水平翻转
                        # print(images[0].size())
                        # image=images[0]#去除了batch_size那一维度，反正batch_size都是1，有没有无所谓
                        batch_size,c,h,w=images.size()
                        if min(h,w) > args.img_size:
                                images= interpolate(images,size=[int(h * (args.img_size / min(h, w))),int(w * (args.img_size / min(h, w)))],mode="bilinear",align_corners=True)
                                # %我就打个比方吧，min(h,w)=h的话  h*(700/min(h,w)=700   w*(700/min(h,w)=w*(700/h)=700*(w/h) 图像的size由[h,w]变为[700,700*(w/h)]
                                #%由此可见，在min(h,w) > 700的前提下，图像被适当的进行分辨率的缩小，到700这一级，但是长宽比是没有改变的，图像没有变形
                                # %这一步操作只是为了对于图像的分辨率的上限进行一个限制
                        batch_size, c, h, w = images.size()
                        #matlab版本的实现中这里是对可能出现的灰度图像进行通道数扩充，并减去图像在各个通道上的均值，以上过程我在dataloader.py中已经实现了，以上。
                        images,labels=images.cuda(),labels.cuda()
                        labels=torch.zeros_like(labels).cuda()
                        # labels=labels.cpu()

                        #传统的SCDA
                        # img_preds, cnnFeature_maps = net(images)  #img_preds:[batch_size=1,200]

                        #Grad_CAM_based_SCDA
                        # grad_cam_scda = GradCam_yuhan_kernel_version30_DGCRL_scda_many_for_retrivial(
                        #     net)  #####!!!!!!!!!!!!!!!!!!!!!最重要的

                        grad_cam_scda =msloss_for_retrieval(net)
                        # grad_cam_scda = PoolCam_yuhan_kernel_for_retrieval(net)

                        ##########################

                        # cam_L28, cam_L31, feature_maps_L28, feature_maps_L31=grad_cam_scda(images)
                        output_mean, output_mean_2,output_mean_3, out_for_visualize,out_for_visualize_kernel = grad_cam_scda(images, ii=ii, flip=flip)
                        # output_mean_26 与output_mean_28 分别存储着来自,'layer4.2.conv2.weight','layer4.1.conv2.weight'的卷积核梯度特征
                        f4_1_1, output_mean_28, f4_1_3, f4_2_1, output_mean_26, f4_2_3 = output_mean[0], output_mean[1], \
                                                                                         output_mean[2], output_mean[3], \
                                                                                         output_mean[4], output_mean[5]
                        # output_mean_26_2 , output_mean_28_2是resnet50的backzone输出的特征图所得到的scda特征，分别是avg_pool与max_pool所对应的特征
                        output_mean_26_2, output_mean_28_2 = output_mean_2[0], output_mean_2[1]
                        embedding_mean_max, embedding_mean_max_1 = output_mean_3[0], output_mean_3[0]

                        feature_maps_L28_mean_norm, feature_maps_L31_mean_norm = output_mean_26, output_mean_28
                        feature_maps_L28_2_mean_norm, feature_maps_L31_2_mean_norm = output_mean_26_2, output_mean_28_2

                        if flip==0:
                            feature_maps=out_for_visualize[0]
                            feature_maps_L31 = feature_maps[
                                0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
                            #
                            feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                                             0)
                            #
                            highlight_conn_L31_beifen=out_for_visualize[1]
                            #
                            feature_maps_L31_sum_noflip=feature_maps_L31_sum
                            #    #这是为最后的可视化准备的
                            #######
                            #######
                            # feature_maps_L31_sum_noflip = feature_maps[0][0][yuri]
                            #

                            highlight_conn_L31_noflip=highlight_conn_L31_beifen



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



tr4_1_1=np.array(tr4_1_1)
tr4_1_3=np.array(tr4_1_3)
tr4_2_1=np.array(tr4_2_1)
tr4_2_3=np.array(tr4_2_3)
te4_1_1=np.array(te4_1_1)
te4_1_3=np.array(te4_1_3)
te4_2_1=np.array(te4_2_1)
te4_2_3=np.array(te4_2_3)

tr_L31_mean_3 = np.array(tr_L31_mean_3)
te_L31_mean_3 = np.array(te_L31_mean_3)

tr_L31_mean_3_1 = np.array(tr_L31_mean_3_1)
te_L31_mean_3_1 = np.array(te_L31_mean_3_1)




# train_data=tr4_1_1.tolist()
# print('train_data.shape:',np.array(train_data).shape)
test_data=te4_1_1.tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
# final_features['train']=train_data
final_features['test']=test_data
layer4_1_conv1_weight=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_1_conv1_weight.json')#'layer4.1.conv1.weight'
# with open(filename,'w') as f_obj:
#     json.dump(final_features,f_obj)



# train_data=tr4_1_3.tolist()
# print('train_data.shape:',np.array(train_data).shape)
test_data=te4_1_3.tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
# final_features['train']=train_data
final_features['test']=test_data
layer4_1_conv3_weight=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_1_conv3_weight.json')#'layer4.1.conv3.weight'
# with open(filename,'w') as f_obj:
#     json.dump(final_features,f_obj)



# train_data=tr4_2_3.tolist()
# print('train_data.shape:',np.array(train_data).shape)
test_data=te4_2_3.tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
# final_features['train']=train_data
final_features['test']=test_data
layer4_2_conv3_weight=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_2_conv3_weight.json')#'layer4.2.conv3.weight'
# with open(filename,'w') as f_obj:
#     json.dump(final_features,f_obj)





# train_data=tr4_2_1.tolist()
# print('train_data.shape:',np.array(train_data).shape)
test_data=te4_2_1.tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
# final_features['train']=train_data
final_features['test']=test_data
layer4_2_conv1_weight=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_2_conv1_weight.json')#'layer4.2.conv1.weight'
# with open(filename,'w') as f_obj:
#     json.dump(final_features,f_obj)







# train_data=tr_L31_mean.tolist()
# print('train_data.shape:',np.array(train_data).shape)
test_data=te_L31_mean.tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
# final_features['train']=train_data
final_features['test']=test_data
layer4_1_conv2_weight=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_1_conv2_weight.json')#'layer4.1.conv2.weight'
# with open(filename,'w') as f_obj:
#     json.dump(final_features,f_obj)




# train_data=tr_L28_mean.tolist()
# print('train_data.shape:',np.array(train_data).shape)
test_data=te_L28_mean.tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
# final_features['train']=train_data
final_features['test']=test_data
layer4_2_conv2_weight=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_2_conv2_weight.json')#'layer4.2.conv2.weight'
# with open(filename,'w') as f_obj:
#     json.dump(final_features,f_obj)













# train_data=np.hstack([tr_L31_mean,
#                       tr_L28_mean,
#                       ]).tolist()
# print('train_data.shape:',np.array(train_data).shape)
test_data=np.hstack([te_L31_mean,
                     te_L28_mean,
                     ]).tolist()
print('test_data.shape:',np.array(test_data).shape)

final_features={}
# final_features['train']=train_data
final_features['test']=test_data
layer4_conv2_weight=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_conv2_weight.json')##'layer4.1.conv2.weight' + #'layer4.2.conv2.weight'
# with open(filename,'w') as f_obj:
#     json.dump(final_features,f_obj)






#
# train_data = tr_L31_mean_2.tolist()
# print('train_data.shape:', np.array(train_data).shape)
test_data = te_L31_mean_2.tolist()
print('test_data.shape:', np.array(test_data).shape)
final_features = {}
# final_features['train'] = train_data
final_features['test'] = test_data
scda_max_avg_norm=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')
# filename = join(target_path, 'scda_max_avg_norm.json')  # 'layer4.1.conv3.weight'
# with open(filename, 'w') as f_obj:
#     json.dump(final_features, f_obj)



#
# train_data = tr_L28_mean_2.tolist()
# print('train_data.shape:', np.array(train_data).shape)
test_data = te_L28_mean_2.tolist()
print('test_data.shape:', np.array(test_data).shape)
final_features = {}
# final_features['train'] = train_data
final_features['test'] = test_data
scda_norm_max_avg=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')
# filename = join(target_path, 'scda_norm_max_avg.json')  # 'layer4.1.conv3.weight'
# with open(filename, 'w') as f_obj:
#     json.dump(final_features, f_obj)


#
#
# train_data = tr_L31_mean_3.tolist()
# print('train_data.shape:', np.array(train_data).shape)
test_data = te_L31_mean_3.tolist()
print('test_data.shape:', np.array(test_data).shape)
final_features = {}
# final_features['train'] = train_data
final_features['test'] = test_data
embedding_max_avg=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')
# filename = join(target_path, 'embedding_max_avg.json')  # 'layer4.1.conv3.weight'
# with open(filename, 'w') as f_obj:
#     json.dump(final_features, f_obj)





# train_data = tr_L31_mean_3_1.tolist()
# print('train_data.shape:', np.array(train_data).shape)
test_data = te_L31_mean_3_1.tolist()
print('test_data.shape:', np.array(test_data).shape)
final_features = {}
# final_features['train'] = train_data
final_features['test'] = test_data
embedding_norm_max_avg=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')
# filename = join(target_path, 'embedding_norm_max_avg.json')  # 'layer4.1.conv3.weight'
#
# with open(filename, 'w') as f_obj:
#     json.dump(final_features, f_obj)







#以下12组任选即可
# test_data=layer4_1_conv2_weight['test']
# print(np.linalg.norm(np.array(test_data)))
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_1_conv2_weight.json')
#
#
# test_data=scda_max_avg_norm['test']
# print(np.linalg.norm(np.array(test_data)))
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/scda_max_avg_norm.json')




# test_data=scda_norm_max_avg['test']
# print(np.linalg.norm(np.array(test_data)))
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/scda_norm_max_avg.json')


# test_data=embedding_max_avg['test']
# print(np.linalg.norm(np.array(test_data)))
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')
#



# test_data=embedding_norm_max_avg['test']
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_norm_max_avg.json')



# test_data=layer4_2_conv3_weight['test']
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_2_conv3_weight.json')


test_data=layer4_conv2_weight['test']
print(np.linalg.norm(np.array(test_data)))
filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_conv2_weight.json')




with open(filename) as f_obj:
    final_features=json.load(f_obj)
train_data=final_features['test']
print(np.array(train_data).shape)



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


train_paths=test_paths
train_labels=test_labels


print(KDTree.valid_metrics)
X = np.array(train_data)
print(np.linalg.norm(X[0]))
query=np.array(test_data)
kdt = KDTree(X, leaf_size=40, metric='l2')
# s = pickle.dumps(kdt)                     # doctest: +SKIP
# kdt = pickle.loads(s)
dis,result=kdt.query(query, k=8000, return_distance=True)
print(dis.shape)
# print(1-0.5*(dis**2))   #向量模为一时
print(1-0.25*(dis**2))    #向量模为二时
print(dis)
#大概就是这样子的，有K个query,最终返回的a矩阵就有K行，每一行都有九个元素，代表
#我们以第一行为例吧，第一行的九个元素。就代表距离第一个query距离最近的九张图片的索引（这九张图片来自训练集），我们通过这个索引
#就可以获知这九张图片的具体的类别（如果我们已知了query的类别，既可以据此来计算mAP），
# 以及对应的存储路径（在图像检索系统中进行最后的检索结果的输出）
print('检索结果的在训练集中的标号索引：')
print(result)
print(result.shape)
query_result_labels=np.zeros(result.shape)
h,w=result.shape
for i in range(h):
    for j in range(w):
        query_result_labels[i,j]=train_labels[result[i,j]]
print('检索结果的类别索引：')
print(query_result_labels)




query_result_paths=[]
h,w=result.shape
for i in range(h):
    query_result_paths.append([])
    for j in range(w):
        query_result_paths[i].append(train_paths[result[i,j]])



# miki_1=cv2.imread(str(test_paths[0]))
miki_1 = Image.open(test_path[0]).convert('RGB')
width,height = miki_1.size
print(miki_1.size)
# print(cam_L31)

print(len(out_for_visualize_kernel))
kernel_0=out_for_visualize_kernel[0]
kernel_1=out_for_visualize_kernel[1]
kernel_2=out_for_visualize_kernel[2]
# kernel_2=torch.from_numpy(output_mean_28_2)
kernel_0 = kernel_0 - torch.min(kernel_0)
kernel_0 = kernel_0 / torch.max(kernel_0)
kernel_0 = kernel_0.cpu().data.numpy()*255*5
kernel_1 = kernel_1 - torch.min(kernel_1)
kernel_1 = kernel_1 / torch.max(kernel_1)
kernel_1 = kernel_1.cpu().data.numpy()*255
kernel_2 = kernel_2 - torch.min(kernel_2)
kernel_2 = kernel_2 / torch.max(kernel_2)
kernel_2 = kernel_2.cpu().data.numpy().reshape(1,kernel_2.shape[0])*255
kernel_2 = kernel_2 * np.ones([512,kernel_2.shape[1]])
cam_L31 = Image.fromarray(kernel_0).convert('L')
print(cam_L31.size)
cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
result = cam_L31*1 #+ miki_1 * 0.5
print(result.shape)
cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/kernel0.jpg'),result)
cv2.waitKey(0)


cam_L31 = Image.fromarray(kernel_1).convert('L')
print(cam_L31.size)
cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
result = cam_L31*1 #+ miki_1 * 0.5
print(result.shape)
cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/kernel1.jpg'),result)
cv2.waitKey(0)


cam_L31 = Image.fromarray(kernel_2).convert('L')
print(cam_L31.size)
cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
result = cam_L31*1 #+ miki_1 * 0.5
print(result.shape)
cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/kernel2.jpg'),result)
cv2.waitKey(0)


highlight_L31=highlight_conn_L31_noflip * 255
# highlight_L28=highlight_conn_L28_noflip * 255
# feature_maps_L28_sum=feature_maps_L28_sum_noflip
feature_maps_L31_sum=feature_maps_L31_sum_noflip
# feature_maps_L28_sum = feature_maps_L28_sum - torch.min(feature_maps_L28_sum)
# feature_maps_L28_sum = feature_maps_L28_sum / torch.max(feature_maps_L28_sum)
feature_maps_L31_sum = feature_maps_L31_sum - torch.min(feature_maps_L31_sum)
feature_maps_L31_sum = feature_maps_L31_sum / torch.max(feature_maps_L31_sum)
feature_maps_L31_sum=feature_maps_L31_sum.cpu().data.numpy()*255
# feature_maps_L28_sum=feature_maps_L28_sum.cpu().data.numpy()*255








cam_L31 = Image.fromarray(highlight_L31).convert('L')
cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
print(cam_L31.size)
# Image._show(cam_L31)
cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
miki_1=cv2.cvtColor(np.asarray(miki_1),cv2.COLOR_RGB2BGR)
# cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
# heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
result = cam_L31 + miki_1 * 0.5
print(result.shape)
# plt.imsave('1.jpg',heatmap_L31)
cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/highlight_L31_original_scda.jpg'),result)
cv2.waitKey(0)

# cam_L31 = Image.fromarray(highlight_L28).convert('L')
# cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
# print(cam_L31.size)
# # Image._show(cam_L31)
# cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
# cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
# miki_1=cv2.cvtColor(np.asarray(miki_1),cv2.COLOR_RGB2BGR)
# # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
# # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
# result = cam_L31*0.9 + miki_1 * 0.2
# print(result.shape)
# # plt.imsave('1.jpg',heatmap_L31)
# cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/highlight_L28_original_scda.jpg'),result)
# cv2.waitKey(0)


cam_L31 = Image.fromarray(feature_maps_L31_sum).convert('L')
cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
print(cam_L31.size)
# Image._show(cam_L31)
cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
miki_1=cv2.cvtColor(np.asarray(miki_1),cv2.COLOR_RGB2BGR)
# cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
# heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
result = cam_L31*1 + miki_1 * 0.5
print(result.shape)
# plt.imsave('1.jpg',heatmap_L31)
cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/feature_maps_L31_sum.jpg'),result)
cv2.waitKey(0)

# cam_L31 = Image.fromarray(feature_maps_L28_sum).convert('L')
# cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
# print(cam_L31.size)
# # Image._show(cam_L31)
# cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
# cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
# miki_1=cv2.cvtColor(np.asarray(miki_1),cv2.COLOR_RGB2BGR)
# # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
# # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
# result = cam_L31*0.9 + miki_1 * 0.2
# print(result.shape)
# # plt.imsave('1.jpg',heatmap_L31)
# cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/feature_maps_L28_sum.jpg'),result)
# cv2.waitKey(0)



plt.figure()
miki=plt.imread(os.path.join(join(args.datasetdir,args.imgdir,img_name)))
plt.imshow(miki)





plt.figure()
for i in range(25):
    plt.subplot(5,5,i+1)
    # print(query_result_paths[0])
    miki=plt.imread(str(query_result_paths[0][i]))
    plt.imshow(miki)
    plt.xticks([])
    plt.yticks([])
plt.show()