import numpy as np
from sklearn.neighbors import KDTree
import json
import matplotlib.pyplot as plt
from SCDA_cars_resnet50.net_res50 import FGIAnet100_metric5_1_for_cam,FGIAnet100_metric6_1_for_cam,\
    FGIAnet100_metric5_for_cam,FGIAnet100_for_cam, FGIAnet100_metric5_2_for_cam
from SCDA_cars_resnet50.bwconncomp import largestConnectComponent
import os
import argparse
from os.path import join
import uuid
import json
from  SCDA_cub_resnet50 import dataloader
from torch.nn.functional import interpolate
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from SCDA_cars_resnet50.grad_cam import GradCam_pp,GradCam,GradCam_new,GradCam_new_many_layers,GradCam_new_many_layers_yu,\
    GradCam_pp_many_layers,GradCam_many_layers,GradCam_new_many_layers_3out,GradCam_new_many_layers_version_2_2out_tmp,\
    GradCam_new_many_layers_version_2,GradCam_new_many_layers_version_2_3out,GradCam_new_many_layers_version_2_3out_tmp,\
    GradCam_new_many_layers_version_2_3out_tmp1,GradCam_new_many_layers_3out0,\
    PoolCam_new_many_layers_3out0,PoolCam_new_many_layers_3out0_saliency_map
import cv2
import random


def random_int(length,a,b):
    list=[]
    count=0
    while (count<length):
        number=random.randint(a,b)
        list.append(number)
        count=count+1
    return list






parser = argparse.ArgumentParser()
parser.add_argument('--datasetdir', default=r'C:\Users\于涵\Desktop\Stanford car dataset\car_ims',  help="path to cub200_2011 dir")
parser.add_argument('--imgdir', default='cars_196',  help="path to train img dir")
parser.add_argument('--tr_te_split_txt', default=r'train_test_split.txt',  help="关于训练集与测试集的划分，0代表测试集，1代表训练集")
parser.add_argument('--tr_te_image_name_txt', default=r'images.txt',  help="关于训练集与测试集的图片的相对路径名字")
parser.add_argument('--image_labels_txt', default=r'image_class_labels.txt',  help="图像的类别标签标记")
parser.add_argument('--class_name_txt', default=r'classes.txt',  help="图像的200个类别名称")
parser.add_argument("--num_classes", type=int, dest="num_classes", help="Total number of epochs to train", default=200)
parser.add_argument('--img_size', type=int, default = 700,help='进行特征提取的图片的尺寸的上界所对应的数量级')
# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_15_1.0.pth')
# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_30_0.9928.pth')
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_90_0.9404.pth')


# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='vgg16-397923af.pth')

parser.add_argument('--savedir',default='./models/', help="Path to save weigths and logs")
args = parser.parse_args()






# net = FGIAnet_3layers_1000()
# net = FGIAnet_vgg_100()#创建该神经网络的一个对象，并将其cuda化    #弱监督检索


net = FGIAnet100_metric5_1_for_cam()
# net = FGIAnet100_metric5_2_for_cam(scale=100)

# net = FGIAnet100_metric6_for_cam()
# net = FGIAnet100_metric5_for_cam()
# net = FGIAnet_2kernels_2layers_for_cam()

# net = FGIAnet_GARD_CAM100_metric()

#SCDA_fine_tuning
# checkpoint = torch.load(join(args.savedir,args.model_name))
checkpoint = torch.load(join(os.path.abspath(os.path.dirname(os.getcwd())),args.savedir,args.model_name))

net.load_state_dict(checkpoint['model'])



#
# # net = FGIAnet_GARD_CAM()
# net = FGIAnet_manylayers()
#
# #SCDA_CAM
# checkpoint = torch.load(join(os.path.abspath(os.path.dirname(os.getcwd())),args.savedir,args.model_name))
# net.load_state_dict(checkpoint['model'])





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

# img_name = r'100FIAT_500_Convertible_2012\008120.jpg'
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
# img_name = r'115Ford_Focus_Sedan_2007\009390.jpg'
# img_name = r'115Ford_Focus_Sedan_2007\009393.jpg'
# img_name =r'111Ford_Ranger_SuperCab_2011\009042.jpg'
# img_name =r'111Ford_Ranger_SuperCab_2011\009084.jpg'
# img_name =r'111Ford_Ranger_SuperCab_2011\009088.jpg'
# img_name  = r'142Infiniti_QX56_SUV_2011\011636.jpg'
# img_name = r'100FIAT_500_Convertible_2012\008135.jpg'
# img_name = r'100FIAT_500_Convertible_2012\008143.jpg'
# img_name = r'100FIAT_500_Convertible_2012\008146.jpg'
# img_name = r'180Spyker_C8_Coupe_2009\014834.jpg'
# img_name= r'160McLaren_MP4-12C_Coupe_2012\013080.jpg'
# img_name=r'16Audi_V8_Sedan_1994\001233.jpg'
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
# img_name=r'85Dodge_Caravan_Minivan_1997\006894.jpg'
# img_name=r'19Audi_TT_Hatchback_2011\001518.jpg'
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

# img_name = r'185Tesla_Model_S_Sedan_2012\015175.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015226.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015218.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015229.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015209.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015243.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015246.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015199.jpg'
# img_name = r'185Tesla_Model_S_Sedan_2012\015207.jpg'

# img_name=r'168Nissan_Leaf_Hatchback_2012\013811.jpg'


# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014496.jpg'
# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014506.jpg'
# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014508.jpg'
# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014502.jpg'
# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014519.jpg'
# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014551.jpg'
# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014543.jpg'
# img_name=r'177Rolls-Royce_Phantom_Sedan_2012\014510.jpg'

# img_name=r'144Jaguar_XK_XKR_2012\011847.jpg'
# img_name = r'149Jeep_Compass_SUV_2012\012206.jpg'

# img_name=r'123Geo_Metro_Convertible_1993\010128.jpg'
# img_name=r'123Geo_Metro_Convertible_1993\010120.jpg'
# img_name=r'123Geo_Metro_Convertible_1993\010129.jpg'
img_name=r'123Geo_Metro_Convertible_1993\010182.jpg'

mikimiki=40

test_path=[]
test_path.append(join(args.datasetdir,args.imgdir,img_name))
loaders = dataloader.get_dataloaders(train_paths, test_path,train_labels,test_labels,args.img_size, 1,1,1,SCDA_flag=1)#返回值为由可迭代DataLoader对象所组成的字典



print("cnn model is ready.")



# miki_1=cv2.imread(str(test_paths[0]))
miki = Image.open(test_path[0]).convert('RGB')
width,height = miki.size
print(miki.size)
# print(cam_L31)


ii=0
for phase in ['test']:
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
                        # labels=labels.cpu()

                        #proxy_based
                        # grad_cam_scda = GradCam_new_many_layers(net)#两个输出，softmax时用，乘上了-1的loss
                        grad_cam_scda = GradCam_new_many_layers_3out(net)#三个输出，度量学习时用,乘上了-1的loss

                        # grad_cam_scda = GradCam_new_many_layers_version_2(net)##两个输出，softmax时用，没有乘上-1的loss
                        grad_cam_scda_2 = GradCam_new_many_layers_version_2_3out(net)##三个输出，度量学习时用,没有乘上-1的loss,特征图点乘梯度
                        grad_cam_scda_2_1 =GradCam_new_many_layers_3out0(net)###三个输出，度量学习时用,没有乘上-1的loss,通道级均值作为权重

                        grad_cam_scda_3 = GradCam_new_many_layers_version_2_3out_tmp(net)##三个输出，度量学习时用，特征图点乘梯度
                        # grad_cam_scda = GradCam_new_many_layers_version_2_2out_tmp(net)##两个输出，softmax时用

                        grad_cam_scda_4 = GradCam_new_many_layers_version_2_3out_tmp1(net)##三个输出，度量学习时用，这是标准的grad_cam，通道级均值作为权重

                        cams ,cams_relu=grad_cam_scda(images)
                        cams_2, cams_relu_2 = grad_cam_scda_2(images)
                        cams_2_1, cams_relu_2_1 = grad_cam_scda_2_1(images)

                        cams_3, cams_relu_3 = grad_cam_scda_3(images,index=mikimiki)
                        cams_4, cams_relu_4 = grad_cam_scda_4(images,index=mikimiki)

# #                        pooling_based_no_saliency_map
#
#                         grad_cam_scda_2_1 = PoolCam_new_many_layers_3out0(net)  ###三个输出，度量学习时用,没有乘上-1的loss,通道级均值作为权重
#                         grad_cam_scda_4 = GradCam_new_many_layers_version_2_3out_tmp1(
#                             net)  ##三个输出，度量学习时用，这是标准的grad_cam，通道级均值作为权重
#
#                         cams_2_1, cams_relu_2_1 = grad_cam_scda_2_1(images)
#                         cams_4, cams_relu_4 = grad_cam_scda_4(images, index=mikimiki)

                      # # # pooling_based_with_saliency_map
                      #
                      #   grad_cam_scda_2_1 = PoolCam_new_many_layers_3out0_saliency_map(net)  ###三个输出，度量学习时用,没有乘上-1的loss,通道级均值作为权重
                      #   grad_cam_scda_4 = GradCam_new_many_layers_version_2_3out_tmp1(
                      #       net)  ##三个输出，度量学习时用，这是标准的grad_cam，通道级均值作为权重
                      #
                      #   cams_2_1, cams_relu_2_1 = grad_cam_scda_2_1(images)
                      #   cams_4, cams_relu_4 = grad_cam_scda_4(images, index=mikimiki)



                        # random_chanels=random_int(10,0,99)
                        # random_chanels=[iii for iii in range(100)]
                        # print(random_chanels)
                        # for j in range(len(random_chanels)):
                        #     cams_random,cams_relu_random = grad_cam_scda_3(images,index=random_chanels[j])
                        #     for i in range(len(cams_random)):
                        #         cam = cams_random[i] * 255
                        #         # print(np.max(cam))
                        #         cam_L31 = Image.fromarray(cam).convert('L')
                        #         cam_L31 = cam_L31.resize((width, height), Image.ANTIALIAS)
                        #         print(cam_L31.size)
                        #         # Image._show(cam_L31)
                        #         cam_L31 = cv2.cvtColor(np.asarray(cam_L31), cv2.COLOR_RGB2BGR)
                        #         # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
                        #         cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
                        #         miki_1 = cv2.cvtColor(np.asarray(miki), cv2.COLOR_RGB2BGR)
                        #         # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
                        #         # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
                        #         result = cam_L31 + miki_1 * 0.3
                        #         # result =  miki_1
                        #
                        #         # result = (cam_L31/255) *miki_1
                        #
                        #         print(result.shape)
                        #         # plt.imsave('1.jpg',heatmap_L31)
                        #         path = './retrivial_visualize/' + 'random_class_'+str(random_chanels[j]) + '_grad_cam.jpg'
                        #         cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())), path), result)
                        #         cv2.waitKey(0)

                            # for i in range(len(cams_relu_random)):
                            #     cam = cams_relu_random[i] * 255
                            #     # print(np.max(cam))
                            #     cam_L31 = Image.fromarray(cam).convert('L')
                            #     cam_L31 = cam_L31.resize((width, height), Image.ANTIALIAS)
                            #     print(cam_L31.size)
                            #     # Image._show(cam_L31)
                            #     cam_L31 = cv2.cvtColor(np.asarray(cam_L31), cv2.COLOR_RGB2BGR)
                            #     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
                            #     cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
                            #     miki_1 = cv2.cvtColor(np.asarray(miki), cv2.COLOR_RGB2BGR)
                            #     # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
                            #     # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
                            #     result = cam_L31 + miki_1 * 0.3
                            #     # result =  miki_1
                            #
                            #     # result = (cam_L31/255) *miki_1
                            #
                            #     print(result.shape)
                            #     # plt.imsave('1.jpg',heatmap_L31)
                            #     path = './retrivial_visualize/' + 'random_chanel_' + str(
                            #         random_chanels[j]) + '_grad_cam_with_relu.jpg'
                            #     cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())), path), result)
                            #     cv2.waitKey(0)

# hina=[0,1,4,6,9,11,13,16,18,20,23,25,27,30]
hina=[29]

# cam_L31=cam_L31_noflip * 255  #因为grad_cam.py中最后是有对cam_L31进行0-1的限制的
# cam_L28=cam_L28_noflip * 255


#注释
# for i  in range(len(cams)):
#     cam=cams[i]*255
#     # print(np.max(cam))
#     cam_L31 = Image.fromarray(cam).convert('L')
#     cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
#     print(cam_L31.size)
#     # Image._show(cam_L31)
#     cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
#     # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     result = cam_L31+ miki_1*0.3
#     # result =  miki_1
#
#     # result = (cam_L31/255) *miki_1
#
#     print(result.shape)
#     # plt.imsave('1.jpg',heatmap_L31)
#     path='./retrivial_visualize/' + str(hina[i]) + '_-1_loss.jpg'
#     cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
#     cv2.waitKey(0)









for i  in range(len(cams_2_1)):
    cam=cams_2_1[i]*255
    # print(np.max(cam))
    cam_L31 = Image.fromarray(cam).convert('L')
    cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
    print(cam_L31.size)
    # Image._show(cam_L31)
    cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
    # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
    cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
    miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
    # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
    # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
    result = cam_L31*1+ miki_1*0.5
    # result =  miki_1

    # result = (cam_L31/255) *miki_1

    print(result.shape)
    # plt.imsave('1.jpg',heatmap_L31)
    path='./retrivial_visualize/' + str(hina[i]) + '_+1_loss_channel_mean.jpg'
    cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
    cv2.waitKey(0)


# for i  in range(len(cams_2)):
#     cam=cams_2[i]*255
#     # print(np.max(cam))
#     cam_L31 = Image.fromarray(cam).convert('L')
#     cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
#     print(cam_L31.size)
#     # Image._show(cam_L31)
#     cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
#     # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     result = cam_L31+ miki_1*0.3
#     # result =  miki_1
#
#     # result = (cam_L31/255) *miki_1
#
#     print(result.shape)
#     # plt.imsave('1.jpg',heatmap_L31)
#     path='./retrivial_visualize/' + str(hina[i]) + '_+1_loss.jpg'
#     cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
#     cv2.waitKey(0)
#
#     # L31_sum_mean = np.mean(cams_2[i])  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
#     # # largestConnectComponent将最大连通区域所对应的像素点置为true
#     # jj = (cams_2[i] > L31_sum_mean)+0
#     #
#     # cam = jj * 255
#     # # print(np.max(cam))
#     # cam_L31 = Image.fromarray(cam).convert('L')
#     # cam_L31 = cam_L31.resize((width, height), Image.ANTIALIAS)
#     # print(cam_L31.size)
#     # # Image._show(cam_L31)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     # cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     # miki_1 = cv2.cvtColor(np.asarray(miki), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     # result = cam_L31 + miki_1 * 0.3
#     # # result =  miki_1
#     #
#     # # result = (cam_L31/255) *miki_1
#     #
#     # print(result.shape)
#     # # plt.imsave('1.jpg',heatmap_L31)
#     # path = './retrivial_visualize/' + str(hina[i]) + '_+1_loss_highlight.jpg'
#     # cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())), path), result)
#     # cv2.waitKey(0)
#



# for i  in range(len(cams_3)):
#     cam=cams_3[i]*255
#     # print(np.max(cam))
#     cam_L31 = Image.fromarray(cam).convert('L')
#     cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
#     print(cam_L31.size)
#     # Image._show(cam_L31)
#     cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
#     # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     result = cam_L31+ miki_1*0.3
#     # result =  miki_1
#
#     # result = (cam_L31/255) *miki_1
#
#     print(result.shape)
#     # plt.imsave('1.jpg',heatmap_L31)
#     path='./retrivial_visualize/' + str(hina[i]) + '_true_grad_cam.jpg'
#     cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
#     cv2.waitKey(0)
#
#     # L31_sum_mean = np.mean(cams_3[i])  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
#     # # largestConnectComponent将最大连通区域所对应的像素点置为true
#     # jj = (cams_3[i] > L31_sum_mean) + 0
#     #
#     # cam = jj * 255
#     # # print(np.max(cam))
#     # cam_L31 = Image.fromarray(cam).convert('L')
#     # cam_L31 = cam_L31.resize((width, height), Image.ANTIALIAS)
#     # print(cam_L31.size)
#     # # Image._show(cam_L31)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     # cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     # miki_1 = cv2.cvtColor(np.asarray(miki), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     # result = cam_L31 + miki_1 * 0.3
#     # # result =  miki_1
#     #
#     # # result = (cam_L31/255) *miki_1
#     #
#     # print(result.shape)
#     # # plt.imsave('1.jpg',heatmap_L31)
#     # path = './retrivial_visualize/' + str(hina[i]) + '_true_grad_cam_highlight.jpg'
#     # cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())), path), result)
#     # cv2.waitKey(0)








for i  in range(len(cams_4)):
    cam=cams_4[i]*255
    # print(np.max(cam))
    cam_L31 = Image.fromarray(cam).convert('L')
    cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
    print(cam_L31.size)
    # Image._show(cam_L31)
    cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
    # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
    cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
    miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
    # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
    # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
    result = cam_L31*1+ miki_1*0.5
    # result =  miki_1

    # result = (cam_L31/255) *miki_1

    print(result.shape)
    # plt.imsave('1.jpg',heatmap_L31)
    path='./retrivial_visualize/' + str(hina[i]) + '_grad_cam_channel_mean.jpg'
    cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
    cv2.waitKey(0)












# for i  in range(len(cams_relu)):
#     cam=cams_relu[i]*255
#     # print(np.max(cam))
#     cam_L31 = Image.fromarray(cam).convert('L')
#     cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
#     print(cam_L31.size)
#     # Image._show(cam_L31)
#     cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
#     # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     result = cam_L31+ miki_1*0.3
#     # result =  miki_1
#
#     # result = (cam_L31/255) *miki_1
#
#     print(result.shape)
#     # plt.imsave('1.jpg',heatmap_L31)
#     path='./retrivial_visualize/' + str(hina[i]) + '_-1_loss_with_relu.jpg'
#     cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
#     cv2.waitKey(0)






for i  in range(len(cams_relu_2_1)):
    cam=cams_relu_2_1[i]*255
    # print(np.max(cam))
    cam_L31 = Image.fromarray(cam).convert('L')
    cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
    print(cam_L31.size)
    # Image._show(cam_L31)
    cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
    # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
    cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
    miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
    # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
    # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
    result = cam_L31*1+ miki_1*0.5
    # result =  miki_1

    # result = (cam_L31/255) *miki_1

    print(result.shape)
    # plt.imsave('1.jpg',heatmap_L31)
    path='./retrivial_visualize/' + str(hina[i]) + '_+1_loss_with_relu_channel_mean.jpg'
    cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
    cv2.waitKey(0)








# for i  in range(len(cams_relu_2)):
#     cam=cams_relu_2[i]*255
#     # print(np.max(cam))
#     cam_L31 = Image.fromarray(cam).convert('L')
#     cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
#     print(cam_L31.size)
#     # Image._show(cam_L31)
#     cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
#     # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     result = cam_L31+ miki_1*0.3
#     # result =  miki_1
#
#     # result = (cam_L31/255) *miki_1
#
#     print(result.shape)
#     # plt.imsave('1.jpg',heatmap_L31)
#     path='./retrivial_visualize/' + str(hina[i]) + '_+1_loss_with_relu.jpg'
#     cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
#     cv2.waitKey(0)
#
#     # L31_sum_mean = np.mean(cams_relu_2[i])  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
#     # # largestConnectComponent将最大连通区域所对应的像素点置为true
#     # jj = (cams_relu_2[i] > L31_sum_mean) + 0
#     #
#     # cam = jj * 255
#     # # print(np.max(cam))
#     # cam_L31 = Image.fromarray(cam).convert('L')
#     # cam_L31 = cam_L31.resize((width, height), Image.ANTIALIAS)
#     # print(cam_L31.size)
#     # # Image._show(cam_L31)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     # cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     # miki_1 = cv2.cvtColor(np.asarray(miki), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     # result = cam_L31 + miki_1 * 0.3
#     # # result =  miki_1
#     #
#     # # result = (cam_L31/255) *miki_1
#     #
#     # print(result.shape)
#     # # plt.imsave('1.jpg',heatmap_L31)
#     # path = './retrivial_visualize/' + str(hina[i]) + '_+1_loss_with_relu_highlight.jpg'
#     # cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())), path), result)
#     # cv2.waitKey(0)

# for i  in range(len(cams_relu_3)):
#     cam=cams_relu_3[i]*255
#     # print(np.max(cam))
#     cam_L31 = Image.fromarray(cam).convert('L')
#     cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
#     print(cam_L31.size)
#     # Image._show(cam_L31)
#     cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
#     # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     result = cam_L31+ miki_1*0.3
#     # result =  miki_1
#
#     # result = (cam_L31/255) *miki_1
#
#     print(result.shape)
#     # plt.imsave('1.jpg',heatmap_L31)
#     path='./retrivial_visualize/' + str(hina[i]) + '_true_grad_cam_with_relu.jpg'
#     cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
#     cv2.waitKey(0)
#
#     # L31_sum_mean = np.mean(cams_relu_3[i])  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
#     # # largestConnectComponent将最大连通区域所对应的像素点置为true
#     # jj = (cams_relu_3[i] > L31_sum_mean) + 0
#     #
#     # cam = jj * 255
#     # # print(np.max(cam))
#     # cam_L31 = Image.fromarray(cam).convert('L')
#     # cam_L31 = cam_L31.resize((width, height), Image.ANTIALIAS)
#     # print(cam_L31.size)
#     # # Image._show(cam_L31)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     # cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     # miki_1 = cv2.cvtColor(np.asarray(miki), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     # result = cam_L31 + miki_1 * 0.3
#     # # result =  miki_1
#     #
#     # # result = (cam_L31/255) *miki_1
#     #
#     # print(result.shape)
#     # # plt.imsave('1.jpg',heatmap_L31)
#     # path = './retrivial_visualize/' + str(hina[i]) + '_true_grad_cam_with_relu_highlight.jpg'
#     # cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())), path), result)
#     # cv2.waitKey(0)





for i  in range(len(cams_relu_4)):
    cam=cams_relu_4[i]*255
    # print(np.max(cam))
    cam_L31 = Image.fromarray(cam).convert('L')
    cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
    print(cam_L31.size)
    # Image._show(cam_L31)
    cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
    # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
    cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
    miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
    # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
    # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
    result = cam_L31*1+ miki_1*0.5
    # result =  miki_1

    # result = (cam_L31/255) *miki_1

    print(result.shape)
    # plt.imsave('1.jpg',heatmap_L31)
    path='./retrivial_visualize/' + str(hina[i]) + '_grad_cam_with_relu_channel_mean.jpg'
    cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
    cv2.waitKey(0)











for i  in range(len(cams_relu_4)):
    cam=cams_relu_4[i]*255
    # print(np.max(cam))
    cam_L31 = Image.fromarray(cam).convert('L')
    cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
    print(cam_L31.size)
    # Image._show(cam_L31)
    cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
    # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
    cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
    miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
    # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
    # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
    result = cam_L31*0+ miki_1*1
    # result =  miki_1

    # result = (cam_L31/255) *miki_1

    print(result.shape)
    # plt.imsave('1.jpg',heatmap_L31)
    path='./retrivial_visualize/' + 'original_picture.jpg'
    cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
    cv2.waitKey(0)










#
# cam_L31 = Image.fromarray(cam_L28).convert('L')
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
# cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/WACD_L28.jpg'),result)
# cv2.waitKey(0)








