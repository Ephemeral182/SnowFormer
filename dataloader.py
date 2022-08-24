from distutils.command.clean import clean
import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
import random
from PIL import Image
from torchvision.utils import make_grid
#from RandomMask1 import *
random.seed(2)
np.random.seed(2)

p = 1
AugDict = {
    1:tfs.ColorJitter(brightness=p),  #Brightness
    2:tfs.ColorJitter(contrast=p), #Contrast
    3:tfs.ColorJitter(saturation=p), #Saturation
    4:tfs.GaussianBlur(kernel_size=5), #Gaussian Blur
    #5:GaussianNoise(std=1), #Gaussian Noise
    #5:RandomMaskwithRatio(64,patch_size=4,ratio=0.7), #Random Mask
}

class CSD_Dataset(data.Dataset):
    def __init__(self,path,train=False,size=256,format='.tif',rand_inpaint=False,rand_augment=None):
        super(CSD_Dataset,self).__init__()
        self.size=size
        self.rand_augment=rand_augment
        self.rand_inpaint=rand_inpaint
        self.InpaintSize = 64
        print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'Snow'))
        print('======>total number for training:',len(self.haze_imgs_dir))
        self.haze_imgs=[os.path.join(path,'Snow',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'Gt')
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        self.format = self.haze_imgs[index].split('/')[-1].split(".")[-1]
        while haze.size[0]<self.size or haze.size[1]<self.size :
            if isinstance(self.size,int):
                index=random.randint(0,10000)
                haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1].split(".")[0]
        clear_name=id+'.'+self.format
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str) and self.train:
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB"))
        return haze,clear,id
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)

        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        
        return  data,target
    def __len__(self):
        return len(self.haze_imgs)


class SRRS_Dataset(data.Dataset):
    def __init__(self,path,train=False,size=256,format='.tif',rand_inpaint=False,rand_augment=None):
        super(SRRS_Dataset,self).__init__()
        self.size=size
        self.rand_augment=rand_augment
        self.rand_inpaint=rand_inpaint
        self.InpaintSize = 64
        print('crop size',size)
        self.train=train
        self.format=format
        if self.train:
            self.haze_imgs_dir=os.listdir(os.path.join(path,'Syn'))
        else:
            self.haze_imgs_dir=os.listdir(os.path.join(path,'Syn'))
        #self.haze_imgs_dir.sort()
        print('======>total number for training:',len(self.haze_imgs_dir))
        self.haze_imgs=[os.path.join(path,'Syn',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'gt')
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        self.format = self.haze_imgs[index].split('/')[-1].split(".")[-1]
        while haze.size[0]<self.size or haze.size[1]<self.size :
            if isinstance(self.size,int):
                index=random.randint(0,10000)
                haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1].split(".")[0]
        clear_name=id+'.'+'jpg'
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str) and self.train:
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB"))
        return haze,clear,id
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)

        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        
        return  data,target
    def __len__(self):
        return len(self.haze_imgs)

class Snow100K_Dataset(data.Dataset):
    def __init__(self,path,train=False,size=256,format='.tif',rand_inpaint=False,rand_augment=None):
        super(Snow100K_Dataset,self).__init__()
        self.size=size
        self.rand_augment=rand_augment
        self.rand_inpaint=rand_inpaint
        self.InpaintSize = 64
        print('crop size',size)
        self.train=train
        #if self.train:
        self.format=format
        if self.train:
            self.haze_imgs_dir=os.listdir(os.path.join(path,'synthetic'))
        else:
            self.haze_imgs_dir=os.listdir(os.path.join(path,'synthetic'))
        #self.haze_imgs_dir.sort()
        print('======>total number for training:',len(self.haze_imgs_dir))
        self.haze_imgs=[os.path.join(path,'synthetic',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'gt')
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        self.format = self.haze_imgs[index].split('/')[-1].split(".")[-1]
        while haze.size[0]<self.size or haze.size[1]<self.size :
            if isinstance(self.size,int):
                index=random.randint(0,10000)
                haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1].split(".")[0]
        clear_name=id+'.'+ self.format
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str) and self.train:
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)

        haze,clear=self.augData(haze.convert("RGB"),clear.convert("RGB"))
        return haze,clear,id
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)

        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        
        return  data,target
    def __len__(self):
        return len(self.haze_imgs)
