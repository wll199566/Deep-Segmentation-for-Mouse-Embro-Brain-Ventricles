
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import random
import pathlib
import nibabel as nib
from skimage import measure

from Extract_mouse_data import Mouse_sub_volumes
from data_augmentation import Rotate, Flip
from seg_net import Net


# In[4]:


#####Function and class here
def get_subbox_idx(input_idx,input_all_data):
    sub_box=[]
    pos_num=0
    for idx in input_idx:
        for i in range(len(input_all_data[idx][0])):
            sub_box.append((idx,input_all_data[idx][0][i]))
        pos_num+=len(input_all_data[idx][0])
    print(pos_num)
    print(len(sub_box))
    return sub_box

def save_nii(img, pred, lbl, save_num, score):
    img_nft = nib.Nifti1Image(np.squeeze(img),np.eye(4))
    pred_nft = nib.Nifti1Image(np.squeeze(pred), np.eye(4))
    lbl_nft = nib.Nifti1Image(np.squeeze(lbl), np.eye(4))
    
    nib.save(img_nft, './predict/img{}.nii'.format(save_num))
    nib.save(pred_nft,'./predict/pred{}_{:.3f}.nii'.format(save_num,score))
    nib.save(lbl_nft, './predict/label{}.nii'.format(save_num))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for i_batch, sample_batched in enumerate(train_loader):
        inputs, labels = sample_batched['image'], sample_batched['label'] 
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        
        del inputs
        
        loss = criterion(outputs, labels.float())
        
        del labels 
        del sample_batched
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        del loss
        print_interval = 1
        if i_batch % print_interval == 0:
            print("epoch {}, batch {}, current loss {}".format(epoch+1,i_batch,running_loss/print_interval))
            running_loss = 0.0

    
def test(model1, model2, device, full_img, full_fil_img, full_label, right_corner, save_num, box_size=128, save_thresh = 0.92):
    model1.eval()
    model2.eval()
    with torch.no_grad():
        input_img = np.zeros((1,1,box_size,box_size,box_size), np.float32)
        input_img[0,0,...] = full_img[right_corner[0]-box_size:right_corner[0],
                                      right_corner[1]-box_size:right_corner[1],
                                      right_corner[2]-box_size:right_corner[2]]
        #input_img[0,1,...] = full_fil_img[right_corner[0]-box_size:right_corner[0],
        #                                  right_corner[1]-box_size:right_corner[1],
        #                                  right_corner[2]-box_size:right_corner[2]]
        
        y_hat1 = model1(torch.from_numpy(input_img).to(device))
        #y_hat2 = model2(torch.from_numpy(input_img).to(device))
        y_hat1[y_hat1 >  save_thresh] = 1.0
        y_hat1[y_hat1 <= save_thresh] = 0.0
        #y_hat2[y_hat2 >  save_thresh] = 1.0
        #y_hat2[y_hat2 <= save_thresh] = 0.0
        y_hat = y_hat1 #+ y_hat2
        y_hat[y_hat > 0] = 1.0
        y_predict = np.zeros((np.shape(full_img)),np.uint8)
        y_predict[right_corner[0]-box_size:right_corner[0],
                  right_corner[1]-box_size:right_corner[1],
                  right_corner[2]-box_size:right_corner[2]] = np.squeeze(y_hat.cpu().numpy())
        #y_predict_component = measure.label(y_predict)
        #component_num = np.unique(y_predict_component)
        #for current_component in range(1,len(component_num)):
        #    if np.sum(y_predict_component == current_component) < 300:
        #        y_predict[y_predict_component == current_component] = 0
        # score binary output
        score = 2*np.sum(y_predict*full_label) / (np.sum(y_predict) + np.sum(full_label))
        save_nii(full_img + 0.5, y_predict, full_label, save_num, score)
        #save_nii(input_img[0,0,...] + 0.5, y_hat.cpu().numpy(), full_label[right_corner[0]-box_size:right_corner[0],
        #                              right_corner[1]-box_size:right_corner[1],
        #                              right_corner[2]-box_size:right_corner[2]], save_num, score)
        contain_ratio = (np.sum(full_label[right_corner[0]-box_size:right_corner[0],
                                          right_corner[1]-box_size:right_corner[1],
                                          right_corner[2]-box_size:right_corner[2]]) / (np.sum(full_label) + 0.0001))
        
        print('Img_num: {}, contain_ratio: {}, f-score: {}'.format(save_num, contain_ratio, score))
        print('predict_bv_pixel: {}, true_bv_pixel: {}'.format(np.sum(y_predict),np.sum(full_label)))
        del input_img, y_predict
    return score

def test_ensemble(model_list, device, test_loader):
    for i in range(len(model_list)):
        model_list[i].eval()
    #model3.eval()
    correct_num = 0
    total_num = 0
    positive_correct=0
    positive_num=0
    negative_correct=0
    negative_num=0
    
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            inputs, labels = sample_batched['image'], sample_batched['label']  
            inputs = inputs.to(device)
            # forward + backward + optimize
            outputs = torch.nn.functional.softmax(model_list[0](inputs),dim=1)
            for i in range(1,len(model_list)):
                outputs += torch.nn.functional.softmax(model_list[i](inputs),dim=1)
            _, predicted = torch.max(outputs, 1)
            correct_num+=np.sum(predicted.cpu().numpy()==labels.numpy())
            total_num+=len(labels)
            positive_correct+=np.sum(predicted.cpu().numpy()*labels.numpy())
            positive_num+=np.sum(labels.numpy())
            negative_correct+=np.sum((1-predicted.cpu().numpy())*(1-labels.numpy()))
            negative_num+=np.sum(1-labels.numpy())
            
    print('total_num:{}, test accuracy:{}, positive_acc:{}, negative_acc:{}'.format(total_num,
                                                                                   correct_num/total_num,
                                                                                    positive_correct/positive_num,
                                                                                    negative_correct/negative_num
                                                                                    ))




def load_data(data_path):
    with open(data_path,'rb') as f:
        #(pos_subvolumes,img2,filtered_img2,img_label2,data_dic[i][0])
        all_data = pickle.load(f)
    f.close()
    return all_data

def dice_loss(source, target):
    # flatten images to vectors so score can be computed with vector ops 
    batch_size, num_channels = source.size(0), source.size(1)
    
    smooth = 1e-4
    s = source.view(batch_size, -1)
    t = target.view(batch_size, -1)
    
    # positive class Dice
    intersect   =     (s * t).sum(dim=1)
    cardinality =     s.sum(dim=1) + t.sum(dim=1)
    
    dsc_p = (2*intersect+smooth)/(cardinality+smooth)
    
    return 1-dsc_p.mean()


# In[5]:


data_path = os.path.join('/scratch/zq415/grammar_cor/Localization','data','test_predict_c1_v2_onemodel.pickle')
#(all_data[current_idx][5],all_data[current_idx][6],all_data[current_idx][7],x2,y2,z2)
all_data = load_data(data_path)
HALF_SIDE = 128 # half of the bounding size 128

all_whole_volumes = {}
all_whole_filtered_volumes = {}
all_whole_labels = {}
all_right_corner = {}
for i in range(len(all_data)):
    all_whole_volumes[i] = all_data[i][0] - 0.5 
    all_whole_filtered_volumes[i] = all_data[i][1] - 0.5
    all_whole_labels[i] = all_data[i][2]
    all_right_corner[i] = (all_data[i][3],all_data[i][4],all_data[i][5])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net1 = Net()
net2 = Net()

#net.apply(weight_init)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net1 = nn.DataParallel(net1)
    net2 = nn.DataParallel(net2)
net1.to(device)
net2.to(device)


print("There are {} parameters in the model".format(count_parameters(net1)))
net1.load_state_dict(torch.load('./model/bv_seg_net128_c1_b4_e5_v1_22.pth'))
net2.load_state_dict(torch.load('./model/bv_seg_net128_c1_b4_e4_v1_22.pth'))

#test
test_result = []
bad_result = []
for i in range(len(all_data)):
    test_result.append(test(net1,net2, device, all_whole_volumes[i], all_whole_filtered_volumes[i], all_whole_labels[i],
         all_right_corner[i], i, box_size=128, save_thresh = 0.92))
    if test_result[i] < 0.6:
        bad_result.append((i, test_result[i]))
result = 0.0
count = 0
test_result_111 = []
for i in range(len(test_result)):
    if test_result[i] > 0.6:
        result += test_result[i]
        test_result_111.append(test_result[i])
        count += 1
print('average_dice: {}'.format(result/count))
print('bad img: {}'.format(bad_result))
print(len(test_result_111))
print('average dice: {}, std: {}'.format(np.mean(test_result_111),np.std(test_result_111)))

