from torch import nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__() 
        self.layer1 = nn.Sequential(
        
            torch.nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.dense_layer = nn.Sequential(

            nn.Linear(in_features=110592,out_features=10),
            nn.Tanh()
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
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN,self).__init__() # Initializing content from the parent class
#         self.cnn_model = nn.Sequential(
#             nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,padding='same'),
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2,stride =1),

#             nn.Conv2d(in_channels=6,out_channels=16,kernel_size=3,padding='same'),
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2,stride =1),
        

#             # nn.Conv2d(kernel_size=3,padding='same'),
#             # nn.Tanh(),
#             # nn.avg_pool2d(kernel_size=2,stride =1),        
        

#             # nn.Conv2d(kernel_size=3,padding='same'),
#             # nn.Tanh(),
#             # nn.avg_pool2d(kernel_size=2,stride =1),


#             # nn.Conv2d(kernel_size=3,padding='same'),
#             # nn.Tanh(),
#             # nn.avg_pool2d(kernel_size=2,stride =1),

#         )
    

#         self.dense_layer = nn.Sequential(

#             nn.Linear(in_features=16,out_features=10),
#             nn.Tanh()
#         )

#     def forward(self,x):
#         x = self.cnn_model(x)
#         x = x.view(x.size(0),-1)
#         x = self.dense_layer(x)
#         x = F.softmax(x)

#         return x


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.layer1 = torch.nn.Sequential(
#             torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#             # torch.nn.Dropout(p=1 - keep_prob))
#         self.layer2 = torch.nn.Sequential(
#             torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#             # torch.nn.Dropout(p=1 - keep_prob))
#         self.layer3 = torch.nn.Sequential(
#             torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
#             # torch.nn.Dropout(p=1 - keep_prob))
#         torch.nn.init.xavier_uniform(self.fc1.weight)
#         self.layer4 = torch.nn.Sequential(
#             self.fc1,
#             torch.nn.ReLU())
#             # torch.nn.Dropout(p=1 - keep_prob))
#         self.fc2 = torch.nn.Linear(625, 10, bias=True)
#         torch.nn.init.xavier_uniform_(self.fc2.weight) 
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         out = self.fc2(out)
#         return out
