from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from os import listdir, path
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from about_msloss.PK_sample import RandomIdentitySampler
from collections import defaultdict

mean = (0.485, 0.456, 0.406)
# std = (1, 1, 1)#这是归一化之后的数值
std=(0.229, 0.224, 0.225)
# mean = (123.6800, 116.7790, 103.9390)

# mean = [104. / 255, 117. / 255, 128. / 255]
# std = 3 * [1. / 255]   #对应bninception

# mean,std=[0.4707, 0.4601, 0.4549], [0.2767, 0.2760, 0.2850]
#CLASS torch.utils.data.Dataset
#An abstract class representing a Dataset.
#All datasets that represent a map from keys to data samples should subclass it. All subclasses should overrite __getitem__(),
#supporting fetching a data sample for a given key. Subclasses could also optionally overwrite __len__(),
#which is expected to return the size of the dataset by many Sampler implementations and the default options of DataLoader.
#https://blog.csdn.net/chituozha5528/article/details/78354833 关于__getitem__()
#https://www.jianshu.com/p/3c6abf767e45   关于__len__()
#https://www.jianshu.com/p/bf2215d9cfe4    关于os.listdir()
#https://blog.csdn.net/leemboy/article/details/83792729    关于PIL
class FGIADataset(Dataset):
    def __init__(self, image_paths,labels,transform=None):
        """
        Args:
            image_paths: A list of paths of all the images including train or test dataset,this assume that you have split the cub200 into train and test
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.images_path = image_paths
        self.labels=labels
        self.label_index_dict = self._build_label_index_dict()

    def __len__(self):
        return len(self.images_path)#图片总量，通过len(class DenoisingDataset(Dataset)类的一个对象)调用

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.labels):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, idx):#这样一种调用格式，假设class DenoisingDataset类的一个对象为miki,miki(idx)则调用该方法，完成对于具体某张图片的操作
        # mean = (123.6800, 116.7790, 103.9390)
        img_name = self.images_path[idx]
        label=self.labels[idx]
        # image = Image.open(img_name).convert('L')#打开读入图片并将其模式转化为灰度图像
        image = Image.open(img_name).convert('RGB')#打开读入图片

        # r, g, b = image.split()
        # image = Image.merge("RGB", (b, g, r))#如果使用bninception作为backbone,请恢复这两行注释

        # image.show()
        # mean = torch.tensor(mean)
        # mean = mean.reshape(3, 1, 1)
        # mean = mean.expand(3, 14, 14)
        # image=image-mean
##################################3
        # plt.imshow(image)
        # plt.show()
        ###################################33
        if self.transform:
            image = self.transform(image)#一系列转换的集合，见下文

        # T=transforms.ToTensor()
        # image=T(image)

        # if image.shape[0]!=3:
        #     image = image.expand(3, image.shape[-2], image.shape[-1])
        normalize=transforms.Normalize(mean=mean, std=std)
##############################################
        # c, h, w = image.size()
        # img = image.numpy()#.astype('uint8')
        # # img = img.reshape(h, w, c)
        # img = np.transpose(img, (1, 2, 0))
        # print(img.shape)
        # cv2.imshow(' ',img)
        # cv2.waitKey(0)
        #################################33333
        image=normalize(image)

        # c, h, w = image.size()
        # img = (image.cpu().data.numpy() * 255).astype('uint8')
        # img = img.reshape(h, w, c)
        # # print(img.shape)
        # cv2.imshow(' ', img)
        # cv2.waitKey(0)
        return image,label
        # return image,label


def collate_fn(batch):
    imgs, labels = zip(*batch)
    labels = [int(k) for k in labels]
    labels = torch.tensor(labels, dtype=torch.int64)
    return torch.stack(imgs, dim=0), labels



def get_dataloaders(train_path_list, test_path_list,train_labels,test_labels,resize_img_size=256, train_batch_size=1,test_batch_size=1,flag=0,SCDA_flag=0):
    batch_sizes = {'train': train_batch_size, 'test':test_batch_size}
    if SCDA_flag==0:
        train_transforms = transforms.Compose([
            transforms.Resize(size=resize_img_size),
            transforms.RandomResizedCrop(scale= [0.16, 1],size = 227),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.ToTensor()
            # transforms.Normalize(mean=mean, std=std)
        ])#Convert a PIL Image or numpy.ndarray to tensor.
            #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            #if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
    else:
        train_transforms = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(227),
            transforms.ToTensor()
            # transforms.Normalize(mean=mean, std=std)
        ])  # Convert a PIL Image or numpy.ndarray to tensor.
    if SCDA_flag == 0:
        test_transforms = transforms.Compose([
            # transforms.Resize(size=(resize_img_size, resize_img_size)),
            transforms.Resize(size=256),
            transforms.CenterCrop(227),
            transforms.ToTensor()
            # transforms.Normalize(mean=mean,std=std)
        ])
    else:
        print('yu_msloss/////')
        test_transforms = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(227),
            transforms.ToTensor()
        ])

    data_transforms = {'train': train_transforms,
                       'test': test_transforms}
    image_datasets = {'train': FGIADataset(train_path_list,train_labels, data_transforms['train']),
                      'test': FGIADataset(test_path_list, test_labels,data_transforms['test'])}

    sampler = {x : RandomIdentitySampler(dataset=image_datasets[x],
                                    batch_size=batch_sizes[x],
                                    num_instances=5,
                                    max_iters=3000
                                    ) for x in ['train']}

    if flag==0:
        pass
        # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x], shuffle=False) for x in ['test']}#shuffle为True其实也可以，就是说处理的顺序打乱了
    # dataloaders['train'] = { torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_sizes['train'], shuffle=True)}
    elif flag==1:
        if SCDA_flag == 0:
            dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'],
                                                                batch_sampler=sampler['train'],
                                                                collate_fn=collate_fn,
                                                                pin_memory=True),
                           'test':torch.utils.data.DataLoader(image_datasets['test'],
                                                              batch_size=batch_sizes['test'],
                                                              shuffle=False,
                                                              collate_fn=collate_fn)}#shuffle为True其实也可以，就是说处理的顺序打乱了
        else:
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                          batch_size=batch_sizes[x],
                                                          shuffle=False,
                                                          collate_fn=collate_fn)
                           for x in ['test','train']}#shuffle为True其实也可以，就是说处理的顺序打乱了

    return dataloaders
