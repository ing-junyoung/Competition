#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import zipfile
import os
import warnings
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np 
from tqdm import tqdm
import cv2

import os
import timm
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
import time

import logging

import albumentations as A



from datetime import datetime

year = datetime.today().year
month = datetime.today().month
#day = datetime.today().day  

today = str(year) + str(month)  


    
run_info = '{}'.format(today)


if not os.path.exists('checkpoint/{}'.format(run_info)):
    os.mkdir('checkpoint/{}'.format(run_info))

log = logging.getLogger('staining_log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fileHandler = logging.FileHandler('checkpoint/{}/log.txt'.format(run_info))
streamHandler = logging.StreamHandler()
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)
#
log.addHandler(fileHandler)
log.addHandler(streamHandler)  


# Label은 csv file 

train_y = pd.read_csv("train_df.csv")

train_labels = train_y["label"]

label_unique = sorted(np.unique(train_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

train_labels = [label_unique[k] for k in train_labels]



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=88)
        
    def forward(self, x):
        x = self.model(x)
        return x


class Network2(nn.Module):
    def __init__(self):
        super(Network2, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True, num_classes=88)
        
    def forward(self, x):
        x = self.model(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, image_path, mode='test'):
        self.image_path = image_path
        #self.label_path = label_path
        self.mode = mode

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):

        # image
        image = self.image_path[idx]
        
        image = A.Normalize(p=1)(image=image)['image']
        
        if self.mode == 'train':
            label = np.array([self.label_path[idx]])
        
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mode == 'train':
        
            image = A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5)(image=image)['image']
#             image = A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=20, interpolation=0, border_mode=4, p=0.5)(image=image)['image']
            image = A.OneOf([
                              A.GaussianBlur(blur_limit=(1, 3), p=1),
                              A.MedianBlur(blur_limit=3, p=1),
                              A.GaussNoise (var_limit=(10.0, 30.0), p=1)
                              ], p=0.5)(image=image)['image']       

            image = A.VerticalFlip(p=0.5)(image=image)['image']
            image = A.HorizontalFlip(p=0.5)(image=image)['image']
            image = A.RandomRotate90(90, p=0.5)(image=image)['image']
            image = A.CoarseDropout(max_holes=4, max_height=40, max_width=40, 
                                    min_holes=2, min_height=20, min_width=20, p=0.5)(image=image)['image']


            image = transforms.ToTensor()(image)
            label = torch.from_numpy(label)
            
        else:
             # 5-Augmentation
            
            image_vflip = A.VerticalFlip(p=1)(image=image)['image']
            image_vflip = transforms.ToTensor()(image_vflip)
            
            image_hflip = A.HorizontalFlip(p=1)(image=image)['image']  
            image_hflip = transforms.ToTensor()(image_hflip)
            
#             image_rotate = A.RandomRotate90(90, p=1)(image=image)['image']
#             image_rotate = transforms.ToTensor()(image_rotate)
            
    
            image90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            image90 = transforms.ToTensor()(image90)
        
            image180 = cv2.rotate(image, cv2.ROTATE_180)
            image180 = transforms.ToTensor()(image180)
            
            image270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image270 = transforms.ToTensor()(image270)
            
    
            image = transforms.ToTensor()(image)
            #label = torch.from_numpy(label)
        
        
        return {
            'img' : image,
            'img_hflip' : image_hflip,
            'img_vflip' : image_vflip,
            'image90' : image90,
            'image180' : image180,
            'image270' : image270
            
            #'label' : label
        }


all_images = np.load('./test_imgs_384.npy')


testset = CustomDataset(all_images)
batch_size = 16
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, shuffle=False)



folder = '2022531_base_size384_fliprotate_schedulerCosine_shiftscale_allaug++_seed3572_UPSAMPLEFalse30_efficientnet_b4_LSFalse_cutmixFalse_focal_False_mixup_True_normalize'
folder2 = '2022514_base_size600_fliprotate_schedulerCosine_shiftscale_allaug++_seed3572_UPSAMPLEFalse30_tf_efficientnet_b7_ns_LSFalse_cutmixTrue_focal_False'


# Test f1 best 기준  

# model_list = ['checkpoint/{}/fold1/Epoch 155 ACC 0.979 F1 : 0.8881 TEST Loss0.108.pth'.format(folder),
# 'checkpoint/{}/fold2/Epoch 336 ACC 0.978 F1 : 0.8559 TEST Loss0.145.pth'.format(folder),
# 'checkpoint/{}/fold3/Epoch 347 ACC 0.971 F1 : 0.8321 TEST Loss0.190.pth'.format(folder),
# 'checkpoint/{}/fold4/Epoch 452 ACC 0.984 F1 : 0.9256 TEST Loss0.097.pth'.format(folder),
# 'checkpoint/{}/fold5/Epoch 407 ACC 0.980 F1 : 0.9058 TEST Loss0.115.pth'.format(folder)]

TTA = True

test_pred_label = []
space = np.zeros(shape=(len(testset), 88))
device = 'cuda'
for k in range(5):

    
    model = Network().to(device)
    
    #if k >= 5:
        
    #    model = Network2().to(device)
    #    folder = folder2
    #    k -= 5
    
    #    all_images = np.load('test_imgs_600.npy')
    #    testset = CustomDataset(all_images)
    #    batch_size = 16
    #    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2, shuffle=False)
    
    
    
    
    path = sorted(glob('checkpoint/{}/fold{}/*.pth'.format(folder, k+1)))
    f1 = np.array([float(i.split('/')[-1].split('F1')[-1].split('TEST')[0].split(' ')[-2]) for i in path])
    max_idx = np.argmax(f1)
    weight_path = path[max_idx]
    
    
    
    fold_output = []


    model.load_state_dict(torch.load(weight_path))
    model.eval()
        
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            
            print(i+1)

            image = batch['img'].to(device)

            if TTA:
                img_hflip = batch['img_hflip'].to(device)
                img_vflip = batch['img_vflip'].to(device)
                image90 = batch['image90'].to(device)
                image180 = batch['image180'].to(device)
                image270 = batch['image270'].to(device)

                output = model(image)
                output_hflip = model(img_hflip)
                output_vflip = model(img_vflip)
                output90 = model(image90)
                output180 = model(image180)
                output270 = model(image270)

                output = (output + output_hflip + output_vflip + output90 + output180 + output270) / 6
                
                
                for q in range(output.shape[0]):
                    fold_output += [torch.softmax(output, 1).cpu().detach().numpy().tolist()[q]]
                #fold_output.append(torch.softmax(output, 1).cpu().detach())


            else:
                output = model(image)

    space += np.array(fold_output)

space = space/5
test_pred_label += np.argmax(space, 1).tolist()




label_decoder = {val:key for key, val in label_unique.items()}

f_result = [label_decoder[result] for result in test_pred_label]



submission = pd.read_csv("sample_submission.csv")

submission["label"] = f_result




submission.to_csv('512_fliprotatescale_scheduler300_tta4_fold5_allaug_efficientnet_b4_384_cutmix0.6_satu0_scale0.05_normalize.csv' , index = False)

