
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

from Extract_mouse_data import Mouse_sub_volumes
from data_augmentation import Rotate, Flip
from weight_init import weight_init
from my_net import VGG_net


# In[2]:


#####Function and class here
def get_subbox_idx(input_idx,input_all_data):
    sub_box=[]
    neg_num=0
    pos_num=0
    for idx in input_idx:
        for i in range(len(input_all_data[idx][0])):
            sub_box.append((idx,0,input_all_data[idx][0][i]))
        neg_num+=len(input_all_data[idx][0])
        for i in range(len(input_all_data[idx][1])):
            sub_box.append((idx,1,input_all_data[idx][1][i]))
        pos_num+=len(input_all_data[idx][1])
    print(neg_num)
    print(pos_num)
    print(len(sub_box))
    return sub_box

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
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i_batch % 10 == 0:
            print("epoch {}, batch {}, current loss {}".format(epoch+1,i_batch,running_loss/10))
            running_loss = 0.0

def test(model, device, test_loader):
    model.eval()
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
            outputs = model(inputs)
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
    
def test_ensemble(model_list, device, test_loader):
    for model in model_list:
        model.eval()
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
            outputs = model_list[0](inputs)
            for i in range(1,len(model_list)):
                outputs += model_list[i](inputs)
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
        #(neg_subvolumes,pos_subvolumes,img2,filtered_img2,img_label2,data_dic[i][0])
        all_data = pickle.load(f)
    f.close()
    return all_data


# In[ ]:


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch BV classifier.')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=7, metavar='N',
                        help='number of epochs to train (default: 7)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--SGD', action='store_true',
                       help='set optimizer as SGD')
    parser.add_argument('--Adam', action='store_true',
                       help='set optimizer as Adam')
    parser.add_argument('--RMSprop', action='store_true',
                       help='set optimizer as RMSprop')
    parser.add_argument('--save_version', type=int, default=1,
                        help='save_version (default 1)')
    args = parser.parse_args()
    print("batch_size: {}, training epochs: {}, learning rate: {}, SGD {}, Adam {}, RMSprop {}, save_version {}".
         format(args.batch_size, args.epochs, args.lr, args.SGD, args.Adam, args.RMSprop, args.save_version))
    
    data_path = os.path.join(os.getcwd(),'data','2018_0711_train_sub_volumes22.pickle')
    #(neg_subvolumes,pos_subvolumes,img2,filtered_img2, img_label2, 
    # img, filtered_img, img_label, data_dic[i][0])
    all_data = load_data(data_path)
    
    HALF_SIDE = 64 # half of the bounding size 128
    #take 1/6 as test img and the rest as train img
    test_idx = list(range(0,len(all_data),6))
    total_idx = list(range(len(all_data)))
#    train_idx = [x for x in total_idx if x not in test_idx]

    print("test index: {0}".format(test_idx))
    print("train index: {0}".format(total_idx))
    
    all_whole_volumes = {}
    all_whole_filtered_volumes = {}
    for i in range(len(all_data)):
        all_whole_volumes[i] = all_data[i][2] - 0.5
        all_whole_filtered_volumes[i] = all_data[i][3] - 0.5
        
    train_sub_box=get_subbox_idx(total_idx,all_data)
    test_sub_box=get_subbox_idx(test_idx,all_data)

    del all_data
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = VGG_net()
    #net.apply(weight_init)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    print("There are {} parameters in the model".format(count_parameters(net)))
    
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1.2]).to(device))
    if args.SGD == True:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00001)
        print('choose SGD as optimizer')
    elif args.Adam == True:
        optimizer = optim.Adam(net.parameters(), lr=args.lr*10, weight_decay=0.00001)
        print('choose Adam as optimizer')
    elif args.RMSprop == True:
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr*10, weight_decay=0.00001)
        print('choose RMSprop as optimizer')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    for epoch in range(args.epochs):
        scheduler.step()
        Mouse_dataset = Mouse_sub_volumes(all_whole_volumes,all_whole_filtered_volumes,train_sub_box,
                                     transform=transforms.Compose([Rotate(),Flip()]))
        dataloader = DataLoader(Mouse_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)
        train(net, device, dataloader, optimizer, criterion, epoch)
    print('Finished Training')
    torch.save(net.state_dict(), './model/all_bv_classifier64_c1_b{}_e{}_v{}.pth'.format(
                                                                             args.batch_size,
                                                                             args.epochs,
                                                                             args.save_version))
    #####test
    print('train accuracy: ')
    Mouse_dataset = Mouse_sub_volumes(all_whole_volumes,all_whole_filtered_volumes,train_sub_box)
    train_dataloader = DataLoader(Mouse_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)
    test(net, device, train_dataloader)
    
    print('test accuracy: ')
    Mouse_dataset = Mouse_sub_volumes(all_whole_volumes,all_whole_filtered_volumes,test_sub_box)
    test_dataloader = DataLoader(Mouse_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)
    test(net, device, test_dataloader)
if __name__ == '__main__':
    main()
