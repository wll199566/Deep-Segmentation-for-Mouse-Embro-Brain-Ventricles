
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

from Extract_mouse_data import Mouse_sub_volumes
from data_augmentation import Rotate, Flip
from seg_net import Net


# In[2]:


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

def save_nii(img, pred, lbl, i, score):
    img_nft = nib.Nifti1Image(np.squeeze(img),np.eye(4))
    pred_nft = nib.Nifti1Image(np.squeeze(pred), np.eye(4))
    lbl_nft = nib.Nifti1Image(np.squeeze(lbl), np.eye(4))
    
    nib.save(img_nft, './predict/img{}.nii'.format(i))
    nib.save(pred_nft,'./predict/pred{}_{:.3f}.nii'.format(i,score))
    nib.save(lbl_nft, './predict/label{}.nii'.format(i))

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

    
def test(model, device, data_loader, save_thresh = 0.92):
    
    model.eval()
    all_score = np.zeros((len(data_loader),1))
    count = 0
    with torch.no_grad():
        losses = []
        scores = []
        pos_d_scores = []
        neg_d_scores = []
        for i_batch, sample_batched in enumerate(data_loader):
            inputs, labels = sample_batched['image'], sample_batched['label']
            inputs = inputs.to(device)
            y_hat = model(inputs)
            
            y_hat[y_hat >  save_thresh] = 1.0
            y_hat[y_hat <= save_thresh] = 0.0
            print(np.shape(labels.numpy()))
            # score binary output
            score = 2*np.sum(y_hat.cpu().numpy()*labels.numpy()) / (np.sum(y_hat.cpu().numpy())
                                                                 + np.sum(labels.numpy()))
            all_score[count,0] = score
            save_nii(inputs.cpu().numpy()[0,0,...]+0.5, y_hat.cpu().numpy(), labels.numpy(), i_batch, score)
            del inputs, labels, y_hat, sample_batched

            
            print('image {}, f-score: {}'.format(i_batch, score))
        return np.mean(all_score)

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


# In[ ]:

def main():
    parser = argparse.ArgumentParser(description='PyTorch BV segmentation.')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--SGD', action='store_true',
                       help='set optimizer as SGD')
    parser.add_argument('--save_version', type=int, default=1,
                        help='save_version (default 1)')

    args = parser.parse_args()
    print("batch_size: {}, training epochs: {}, learning rate: {}, SGD {}, save_version {}".
         format(args.batch_size, args.epochs, args.lr, args.SGD, args.save_version))

    data_path = os.path.join(os.getcwd(),'data','2018_0711_train_sub_volumes22_seg.pickle')
    #(pos_subvolumes,img,filtered_img,img_label,all_name[i])
    all_data = load_data(data_path)
    HALF_SIDE = 128 # half of the bounding size 128
    #take 1/6 as test img and the rest as train img
    test_idx = list(range(0,len(all_data),6))
    total_idx = list(range(len(all_data)))

    print("test index: {0}".format(test_idx))
    print("train index: {0}".format(total_idx))

    train_sub_box=get_subbox_idx(total_idx,all_data)
    test_sub_box=get_subbox_idx(test_idx,all_data)

    all_whole_volumes = {}
    all_whole_filtered_volumes = {}
    all_whole_labels = {}
    for i in range(len(all_data)):
        all_whole_volumes[i] = all_data[i][1] - 0.5
        #all_whole_filtered_volumes[i] = all_data[i][2] - 0.5
        all_whole_labels[i] = all_data[i][3]
    del all_data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    #net.apply(weight_init)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    print("There are {} parameters in the model".format(count_parameters(net)))


    criterion = dice_loss
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(args.epochs):
        scheduler.step()
        idx1=np.random.choice(range(len(train_sub_box)),22000,replace=False)
        train_data_shuffle =[train_sub_box[idx1[i]] for i in range(len(idx1))]
    
        Mouse_dataset = Mouse_sub_volumes(all_whole_volumes,all_whole_filtered_volumes,all_whole_labels,train_data_shuffle,
                                     transform=transforms.Compose([Rotate(),Flip()]))
        dataloader = DataLoader(Mouse_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)
        train(net, device, dataloader, optimizer, criterion, epoch)

    torch.save(net.state_dict(), './model/bv_seg_net128_c1_b{}_e{}_v{}_22.pth'.format(args.batch_size,args.epochs,args.save_version))

if __name__ == '__main__':
    main()

