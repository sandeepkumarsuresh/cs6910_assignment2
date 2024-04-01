from torch import nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self,kernel_size = 3, conv_stride = 1,conv_pad = 1,pool_kernel_size = 2,pool_stride = 2,in_channel = 3, num_filter = 8,filter_multiplier = 2,dropout = 0.2):
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.conv_pad = conv_pad
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.in_channel = in_channel
        self.filter_multiplier = filter_multiplier
        self.dropout = dropout

        self.layer1_size = int(num_filter)
        self.layer2_size = int(self.layer1_size * filter_multiplier)
        self.layer3_size = int(self.layer2_size * filter_multiplier)
        self.layer4_size = int(self.layer3_size * filter_multiplier)
        self.layer5_size = int(self.layer4_size * filter_multiplier)

        super(CNN,self).__init__() 

        self.layer1 = nn.Sequential(
        
            torch.nn.Conv2d(self.in_channel, self.layer1_size, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.conv_pad),
            nn.BatchNorm2d(self.layer1_size),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(self.layer1_size, self.layer2_size, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.conv_pad),
            nn.BatchNorm2d(self.layer2_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride))
        
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(self.layer2_size, self.layer3_size, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.conv_pad),
            nn.BatchNorm2d(self.layer3_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride))

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(self.layer3_size,self.layer4_size, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.conv_pad),
            nn.BatchNorm2d(self.layer4_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride))

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(self.layer4_size, self.layer5_size, kernel_size=self.kernel_size, stride=self.conv_stride, padding=self.conv_pad),
            nn.BatchNorm2d(self.layer5_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride))
        
        self.dense_layer = nn.Sequential(

            nn.Linear(self.layer5_size * (250 // (self.pool_stride**5)) * (250 // (self.pool_stride**5)),out_features=10),
            nn.Dropout2d(p=self.dropout),
            nn.ReLU()
            
        )

    def forward(self,x):

        l1 = self.layer1(x)
        # print('layer1',l1.shape)
        l2 = self.layer2(l1)
        # print('layer2',l2.shape)
        l3 = self.layer3(l2)
        # print('layer3',l3.shape)
        l4 = self.layer4(l3)
        # print('layer4',l4.shape)
        l5 = self.layer5(l4)
        # print('layer5',l5.shape)
        
        x = torch.flatten(l5, 1)
        # print('flatten shape',x.shape)

        ll = self.dense_layer(x)
        # print(ll.shape)

        return ll
