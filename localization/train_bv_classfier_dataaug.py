
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import random
from torch.utils.data import Dataset, DataLoader
from Extract_mouse_data import Mouse_sub_volumes
from data_augmentation import Rotate, Flip


# In[2]:


data_path = os.path.join(os.getcwd(),'data','2018_0622_all_sub_volumes.pickle')
with open(data_path,'rb') as f:
    #(neg_subvolumes,pos_subvolumes,img2,filtered_img2,img_label2,data_dic[i][0])
    all_data = pickle.load(f)
f.close()


# In[3]:


HALF_SIDE = 64 # half of the bounding size 128
#take 1/6 as test img and the rest as train img
test_idx = list(range(0,len(all_data),6))
total_idx = list(range(len(all_data)))
train_idx = [x for x in total_idx if x not in test_idx]

print("test index: {0}".format(test_idx))
print("train index: {0}".format(train_idx))


# In[4]:


all_whole_volumes = {}
all_whole_filtered_volumes = {}
for i in range(len(all_data)):
    all_whole_volumes[i] = all_data[i][2]
    all_whole_filtered_volumes[i] = all_data[i][3]


# In[5]:


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
train_sub_box=get_subbox_idx(train_idx,all_data)
test_sub_box=get_subbox_idx(test_idx,all_data)


# In[6]:


from vgg_net import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

net.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("There are {} parameters in the model".format(count_parameters(net)))


# In[7]:


import torch.optim as optim
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1.2]).to(device))
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.00001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)


# In[9]:


for epoch in range(7):
    scheduler.step()
    my_batch_size=200
    idx1=np.random.choice(range(len(train_sub_box)),len(train_sub_box)-(len(train_sub_box)%my_batch_size),
                          replace=False)
    
    train_data_shuffle =[train_sub_box[idx1[i]] for i in range(len(idx1))]
    
    Mouse_dataset = Mouse_sub_volumes(all_whole_volumes,all_whole_filtered_volumes,train_data_shuffle,
                                     transform=transforms.Compose([
                                               Rotate(),
                                               Flip()
                                           ]))
    dataloader = DataLoader(Mouse_dataset, batch_size=my_batch_size,
                        shuffle=False, num_workers=4)
    
    running_loss = 0.0
    for i_batch, sample_batched in enumerate(dataloader):
        inputs, labels = sample_batched['image'], sample_batched['label']  
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i_batch % 10 == 0:
            print("epoch {}, batch {}, current loss {}".format(epoch+1,i_batch,running_loss/10))
            running_loss = 0.0
torch.save(net.state_dict(), './model/bv_classify_128_0627_2018_data_aug2.pth')
#torch.save(net, './model/bv_classify_128_0625_2018_data_aug.pth')
print('Finished Training')


# In[ ]:


########for train data
correct_num = 0
total_num = 0
positive_correct=0
positive_num=0
negative_correct=0
negative_num=0
net.eval()

Mouse_dataset = Mouse_sub_volumes(all_whole_volumes,all_whole_filtered_volumes,train_sub_box)
dataloader = DataLoader(Mouse_dataset, batch_size=200,
                        shuffle=False, num_workers=4)
with torch.no_grad():
    for i_batch, sample_batched in enumerate(dataloader):
        inputs, labels = sample_batched['image'], sample_batched['label']
        inputs = inputs.to(device)
        # forward + backward + optimize
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        correct_num+=np.sum(predicted.cpu().numpy()==labels.numpy())
        total_num+=len(labels)
        positive_correct+=np.sum(predicted.cpu().numpy()*labels.numpy())
        positive_num+=np.sum(labels.numpy())
        negative_correct+=np.sum((1-predicted.cpu().numpy())*(1-labels.numpy()))
        negative_num+=np.sum(1-labels.numpy())
            
            
print('total_num:{}, train accuracy:{}, positive_acc:{}, negative_acc:{}'.format(total_num,
                                                                                   correct_num/total_num,
                                                                                    positive_correct/positive_num,
                                                                                    negative_correct/negative_num
                                                                                    ))


# In[ ]:


########for test data
correct_num = 0
total_num = 0
positive_correct=0
positive_num=0
negative_correct=0
negative_num=0
net.eval()

Mouse_dataset = Mouse_sub_volumes(all_whole_volumes,all_whole_filtered_volumes,test_sub_box)
dataloader = DataLoader(Mouse_dataset, batch_size=200,
                        shuffle=False, num_workers=4)
with torch.no_grad():
    for i_batch, sample_batched in enumerate(dataloader):
        inputs, labels = sample_batched['image'], sample_batched['label']  
        inputs = inputs.to(device)
        # forward + backward + optimize
        outputs = net(inputs)
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


