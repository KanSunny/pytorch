
# coding: utf-8

# In[7]:


import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class DogCat(data.Dataset):
    
    # 看输入的是测试集（test）还是训练集，训练集还需划分为训练集（train）和验证集（not train）
    def __init__(self, root, transforms=None, train=True, test=False):
        # 目标：获取所有图片地址，并根据训练、验证、测试划分数据
        
        ####1. 分布加载数据及排序
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]# 读取root中所有图片，并拼接其root
        
        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg
        if self.test:
            # 测试集按照标签排序 root/<num>.jpg
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            # 训练集按照标签排序 root/<category>.<num>.jpg 
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        
        img_num = len(imgs)
        print("总数据量为：%d" % img_num)
        
        ####2. 划分测试、训练、验证数据
        # 划分训练、验证集，训练：验证 = 7:3
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7*img_num)]
        else:
            self.imgs = imgs[int(0.7*img_num):]
        
        ####3. 定义不同的转化方式
        if transforms is None:
            # 数据转换操作，检测验证和训练的数据转换有区别
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            
            # 测试集和验证集
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Scale(224),# 裁剪为224
                    T.CenterCrop(224),# 在中心裁剪
                    T.ToTensor(),
                    normalize# 归一化
                ])
            # 训练集
            else:
                self.transforms = T.Compose([
                    T.Scale(256),# 载256中随机裁剪224
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),# 0.5概率水平翻转，镜面
                    T.ToTensor(),
                    normalize
                ])
    
    def __getitem__(self, index):
        # 返回一张图片的数据，如果是测试集，没有图片id，如1000.jpg返回1000
        
        ####1. 寻找标签
        img_path = self.imgs[index]
        if self.test:
            # 测试集label的标签为序号
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            # 训练集及验证集的标签为dog 1 cat 0
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        
        ####2. 获取数据及转化
        data = Image.open(img_path)
        # 进行适当的转化
        data = self.transforms(data)
        
        return data, label
    
    def __len__(self):
        # 返回数据集中所有图片个数
        
        return len(self.imgs)
        
                              
    

