
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pickle
import os
import random
import nibabel as nib 

from train_bv_classfier_parse import load_data, get_subbox_idx
from my_net import VGG_net
# In[2]:


def extract_subvolumes(whole_img,whole_label,box_size=64,largest_ratio=0.8,step_size=3):
    img_size=np.shape(whole_img)
    bv_voxel_num=np.sum(whole_label)
    sub_volumes=[]
    
    x_start = box_size
    x_stop = img_size[0]
    
    y_start = box_size
    y_stop = img_size[1]
    
    z_start = box_size
    z_stop = img_size[2]
    
    for i in range(x_start,x_stop+1,step_size):
        for j in range(y_start,y_stop+1,step_size):
            for k in range(z_start,z_stop+1,step_size):
                if (np.sum(whole_label[i-box_size:i,
                                j-box_size:j,
                                k-box_size:k])/(bv_voxel_num+0.001)) < largest_ratio:
                    sub_volumes.append((0,(i,j,k)))
                else:
                    sub_volumes.append((1,(i,j,k)))
    return sub_volumes

class Mouse_sub_volumes(Dataset):
    """Mouse sub-volumes BV dataset."""

    def __init__(self, current_whole_volumes, current_whole_filtered_volumes, all_idx, transform=None):
        """
        Args:
            all_whole_volumes: Contain all the padded whole BV volumes as a dic
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.whole_volumes = current_whole_volumes
        self.idx = all_idx
        self.whole_filtered_volumes = current_whole_filtered_volumes
        self.transform = transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, num):
        #idx [0] positive or negative label, [1] x, y, z right corner index
        box_size=64
        label = self.idx[num][0]
        x, y, z = self.idx[num][1]
        img = np.zeros((1,box_size,box_size,box_size),np.float32)
        
        img[0,...] = self.whole_volumes[x-box_size:x,
                                        y-box_size:y,
                                        z-box_size:z]
        
        #img[1,...] = self.whole_filtered_volumes[x-box_size:x,
        #                                        y-box_size:y,
        #                                        z-box_size:z]

        sample = {'image': img, 'label': label, 'x': x, 'y': y, 'z': z}
        return sample
    
def save_nii(img, lbl, i, x, y, z, contain_ratio):
    box_size = 128
    img_nft = nib.Nifti1Image(np.squeeze(img[x-box_size:x, y-box_size:y, z-box_size:z]),np.eye(4))
    lbl_nft = nib.Nifti1Image(np.squeeze(lbl[x-box_size:x, y-box_size:y, z-box_size:z]), np.eye(4))
    
    nib.save(img_nft, './predict/img{}_{}.nii'.format(i, contain_ratio))
    nib.save(lbl_nft, './predict/label{}_{}.nii'.format(i, contain_ratio))
    
def count_contain_ratio(label, box_size, x, y, z):
    return (np.sum(label[x-box_size:x,
                         y-box_size:y,
                         z-box_size:z])/np.sum(label))

def test(net1,net2,net3,current_img, current_fil_img, current_label, threshold):
    net1.eval()
    net2.eval()
    net3.eval()
    correct_num = 0
    total_num = 0
    positive_correct=0
    positive_num=0
    negative_correct=0
    negative_num=0
    predicted_bv_right_corner = []
    true_bv_right_corner = []

    current_samples = extract_subvolumes(current_img, current_label)
    Mouse_dataset = Mouse_sub_volumes(current_img, current_fil_img, current_samples)
    dataloader = DataLoader(Mouse_dataset, batch_size=200, shuffle=False, num_workers=4)
    print("img_size: {}".format(current_img.shape))
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            inputs, labels = sample_batched['image'], sample_batched['label']
            right_corner=[]
            for ii in range(len(sample_batched['label'])):
                right_corner.append((sample_batched['x'][ii],sample_batched['y'][ii],sample_batched['z'][ii]))
            inputs = inputs.to(device)
            # forward + backward + optimize
            outputs = torch.nn.functional.softmax(net1(inputs),dim=1)
            #outputs += torch.nn.functional.softmax(net2(inputs),dim=1)
            #outputs += torch.nn.functional.softmax(net3(inputs),dim=1)
            #outputs /=3.0
            outputs[:,0] += threshold
            _, predicted = torch.max(outputs, 1)
            for ii in range(len(predicted)):
                if predicted[ii]==1:
                    predicted_bv_right_corner.append(right_corner[ii])
            for ii in range(len(labels)):
                if labels[ii]==1:
                    true_bv_right_corner.append(right_corner[ii])
            
            correct_num+=np.sum(predicted.cpu().numpy()==labels.numpy())
            total_num+=len(labels)
            positive_correct+=np.sum(predicted.cpu().numpy()*labels.numpy())
            positive_num+=np.sum(labels.numpy())
            negative_correct+=np.sum((1-predicted.cpu().numpy())*(1-labels.numpy()))
            negative_num+=np.sum(1-labels.numpy())
    return (predicted_bv_right_corner, true_bv_right_corner, correct_num,
            total_num, positive_correct, positive_num, negative_correct, negative_num)


# In[3]:


data_path = os.path.join(os.getcwd(),'data','2018_0711_test_sub_volumes22.pickle')
#(neg_subvolumes,pos_subvolumes,img2,filtered_img2, img_label2, 
# img, filtered_img, img_label, data_dic[i][0])
all_data = load_data(data_path)

HALF_SIDE = 64 # half of the bounding size 128
#take 1/6 as test img and the rest as train img
test_idx = list(range(0,len(all_data),6))
total_idx = list(range(len(all_data)))
print("test index: {0}".format(test_idx))
print("train index: {0}".format(total_idx))

all_whole_volumes = {}
all_whole_filtered_volumes = {}
all_whole_labels = {}
for i in range(len(all_data)):
    all_whole_volumes[i] = all_data[i][2] -0.5
    all_whole_filtered_volumes[i] = all_data[i][3] - 0.5
    all_whole_labels[i] = all_data[i][4]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net1 = VGG_net()
net2 = VGG_net()
net3 = VGG_net()

#net.apply(weight_init)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net1 = nn.DataParallel(net1)
    net2 = nn.DataParallel(net2)
    net3 = nn.DataParallel(net3)
net1.to(device)
net2.to(device)
net3.to(device)
net1.load_state_dict(torch.load('./model/all_bv_classifier64_c1_b200_e7_v1.pth'))
net2.load_state_dict(torch.load('./model/all_bv_classifier64_c1_b200_e6_v1.pth'))
net3.load_state_dict(torch.load('./model/all_bv_classifier64_c1_b200_e5_v1.pth'))


# In[ ]:


########for validation data
count=0

all_predic_bv_right_corner = np.zeros((len(total_idx),3))
all_true_bv_right_corner = np.zeros((len(total_idx),3))
contain_ratio = np.zeros((len(total_idx),2))

wrong_idx = []
test_sub_box = {}

for current_idx in total_idx:            
    (predicted_bv_right_corner, true_bv_right_corner,
     correct_num, total_num, positive_correct, positive_num,
     negative_correct, negative_num) = test(net1, net2, net3, all_whole_volumes[current_idx],
                                            all_whole_filtered_volumes[current_idx],
                                            all_whole_labels[current_idx], 0.9)
    if len(predicted_bv_right_corner) == 0:
        (predicted_bv_right_corner, true_bv_right_corner,
         correct_num, total_num, positive_correct, positive_num,
         negative_correct, negative_num) = test(net1, net2, net3, all_whole_volumes[current_idx],
                                                all_whole_filtered_volumes[current_idx],
                                                all_whole_labels[current_idx], 0.8)
    if len(predicted_bv_right_corner) == 0:
        (predicted_bv_right_corner, true_bv_right_corner,
         correct_num, total_num, positive_correct, positive_num,
         negative_correct, negative_num) = test(net1, net2, net3, all_whole_volumes[current_idx],
                                                all_whole_filtered_volumes[current_idx],
                                                all_whole_labels[current_idx], 0.7)
   
    for ii in range(len(predicted_bv_right_corner)):
        all_predic_bv_right_corner[count,:]+=predicted_bv_right_corner[ii]
    all_predic_bv_right_corner[count,:]/=len(predicted_bv_right_corner)
    
    for ii in range(len(true_bv_right_corner)):
        all_true_bv_right_corner[count,:]+=true_bv_right_corner[ii]
    all_true_bv_right_corner[count,:]/=len(true_bv_right_corner)
    
    x, y, z = (int(all_predic_bv_right_corner[count,0]), 
                    int(all_predic_bv_right_corner[count,1]), 
                    int(all_predic_bv_right_corner[count,2]))
    contain_ratio[count,0] = count_contain_ratio(all_whole_labels[current_idx], HALF_SIDE, x, y, z)
    
    x2, y2, z2 = (int(2*all_predic_bv_right_corner[count,0]), 
                    int(2*all_predic_bv_right_corner[count,1]), 
                    int(2*all_predic_bv_right_corner[count,2]))
    test_sub_box[current_idx] = (all_data[current_idx][5],all_data[current_idx][6],all_data[current_idx][7],
                                x2,y2,z2)
    
    contain_ratio[count,1] = count_contain_ratio(all_data[current_idx][7], HALF_SIDE*2, x2, y2, z2)

    save_nii(all_data[current_idx][5],
             all_data[current_idx][7],
             count+1, x2, y2 ,z2, contain_ratio[count,1])
    if contain_ratio[count,1] < 0.997:
        wrong_idx.append((count,contain_ratio[count,1]))
   
    if negative_num == 0:
        negative_num+=1
    count+=1
    print("count: {}, true_right_corner: {}, predicted_right_corner: {}".format(count,
                                                                        all_true_bv_right_corner[count-1,:], 
                                                         
                                                                                all_predic_bv_right_corner[count-1,:]))
    print("Contain ratio 0: {}, Contain ratio 1: {}".format(contain_ratio[count-1,0],contain_ratio[count-1,1]))
    print('total_num:{}, test accuracy:{}, positive_acc:{}, negative_acc:{}'.format(total_num,
                                                                                   correct_num/total_num,
                                                                                    positive_correct/positive_num,
                                                                                    negative_correct/negative_num
                                                                                    ))
    print("  ")
print(wrong_idx)
save_name = 'test_predict_c1_v2_onemodel.pickle'
save_file = open(os.path.join(os.getcwd(),'data',save_name),'wb')
pickle.dump(test_sub_box,save_file)
save_file.close()

