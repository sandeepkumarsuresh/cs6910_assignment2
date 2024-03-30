import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import os
import cv2
import splitfolders
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from preprocess import Custom_dataset
from CNN import CNN


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':


    # splitfolders.ratio("Datasets/inaturalist_12K/train", output="Train_Val_Dataset",
    #                     seed=1337, ratio=(.8, .2), group_prefix=None, move=False)
    


    train_dataset = Custom_dataset(dataset_path = 'Train_Val_Dataset/train')
    val_dataset = Custom_dataset(dataset_path = 'Train_Val_Dataset/val')


    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Device Available : ",device)

    # cnn_model = CNN()
    layer1 = nn.Sequential(
    
        torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.AvgPool2d(kernel_size=2, stride=2)
    
    )
    layer2 = torch.nn.Sequential(
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2))
    
    for i, data in enumerate(train_dataloader, 0):

        inputs, labels = data
        print(inputs.shape)
        l1 = layer1(inputs)
        # print(l1.shape)
        # l2 = layer2(l1)
        # print(l2.shape)


        break

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)
    # for epoch in range(2):  # loop over the dataset multiple times

    #     running_loss = 0.0
    #     for i, data in enumerate(train_dataloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data

    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = cnn_model(inputs)
    #         print(outputs.shape)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:    # print every 2000 mini-batches
    #             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
    #             running_loss = 0.0

    # print('Finished Training')



    # # Display image and label.
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")