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
from tqdm import tqdm
from preprocess import Custom_dataset
from CNN import CNN



if __name__ == '__main__':

    PATH = './model.pth'

    # splitfolders.ratio("Datasets/inaturalist_12K/train", output="Train_Val_Dataset",
    #                     seed=1337, ratio=(.8, .2), group_prefix=None, move=False)
    


    train_dataset = Custom_dataset(dataset_path = 'Train_Val_Dataset/train')
    val_dataset = Custom_dataset(dataset_path = 'Train_Val_Dataset/val')


    train_dataloader = DataLoader(train_dataset, batch_size=32 ,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Device Available : ",device)

    cnn_model = CNN()

    cnn_model.to(device)



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)

    for epoch in tqdm(range(2)):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data           
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = torch.squeeze(labels)
            # print('label shape',labels.shape)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn_model(inputs)
            # print('forward pass shape',outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(cnn_model.state_dict(), PATH)


    # # Display image and label.
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")