
# coding: utf-8

# In[ ]:
import torch.nn as nn
import torch.nn.functional as F
import torch

class VGG_net(nn.Module):
    def __init__(self,conv_drop_rate=0.15,linear_drop_rate=0.4):
        super(VGG_net, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=22, kernel_size=3,stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm3d(22)
        self.conv2 = nn.Conv3d(in_channels=22, out_channels=22, kernel_size=3,stride=1,padding=1)
        self.conv2_bn = nn.BatchNorm3d(22)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.dropout1 = nn.Dropout3d(conv_drop_rate)
        
        self.conv3 = nn.Conv3d(in_channels=22, out_channels=32, kernel_size=3,stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm3d(32)
        self.conv4 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3,stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.dropout2 = nn.Dropout3d(conv_drop_rate)
        
        self.conv5 = nn.Conv3d(in_channels=32, out_channels=48, kernel_size=3,stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm3d(48)
        self.conv6 = nn.Conv3d(in_channels=48, out_channels=48, kernel_size=3,stride=1, padding=1)
        self.conv6_bn = nn.BatchNorm3d(48)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.dropout3 = nn.Dropout3d(conv_drop_rate)
        
        self.conv7 = nn.Conv3d(in_channels=48, out_channels=64, kernel_size=3,stride=1, padding=1)
        self.conv7_bn = nn.BatchNorm3d(64)
        self.conv8 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3,stride=1, padding=1)
        self.conv8_bn = nn.BatchNorm3d(64)
        self.pool4 = nn.MaxPool3d(2, 2)
        self.dropout4 = nn.Dropout3d(conv_drop_rate)
        
        self.fc1 = nn.Linear(4*4*4*64, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(linear_drop_rate)
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):        
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.dropout1(self.pool1(self.conv2_bn(F.relu(self.conv2(x)))))
        
        x = self.conv3_bn(F.relu(self.conv3(x)))
        x = self.dropout2(self.pool2(self.conv4_bn(F.relu(self.conv4(x)))))
        
        x = self.conv5_bn(F.relu(self.conv5(x)))
        x = self.dropout3(self.pool3(self.conv6_bn(F.relu(self.conv6(x)))))
        
        x = self.conv7_bn(F.relu(self.conv7(x)))
        x = self.dropout4(self.pool4(self.conv8_bn(F.relu(self.conv8(x)))))
        
        x = x.view(-1, 4*4*4*64)
        x = self.dropout5(self.fc1_bn(F.relu(self.fc1(x))))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    pass