# https://blog.paperspace.com/training-validation-and-accuracy-in-pytorch/
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
import wandb

"""
sweep_configuration = {
    'method': 'bayes', #grid, random
    'metric': {
    'name': 'val_Accuracy',
    'goal': 'maximize'   
    },
    'parameters': {
        'filters': {
            'values': [32,64,128]
        },
        'activation_functions': {
            'values': ['RELU','GELU','SiLU']
        },
        'data_augmentation': {
            'values': ['True','False']
        },
        'batch_normalisation': {
            'values': ['True','False']
        },
        'dropout': {
            'values': [0.2,0.3]
        }
    }
}
sweep_id = wandb.sweep(sweep_configuration,project='dl_ass2')


def do_sweep():

    wandb.init()
    config = wandb.config
    run_name = "epochs:"+str(config.epochs)+"hidden_layer:"+str(config.n_hidden_layers)+"_mini_batch_size:"+str(config.batch)+"_activations"+str(config.activation_para)+"loss_function"+str(config.loss_function)+"regularization"+str(config.regularization)
    print(run_name)
    wandb.run.name = run_name

# TODO

"""

def train(model,train_dataloader,optimizer,loss_function):

    model.train() # Tells the model that you are doing the training

    loss_per_batch = []
    for images,labels in train_dataloader:

        images, labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        
        output = model(images)
        labels = torch.squeeze(labels)
        loss = loss_function(output,labels)
        loss_per_batch.append(loss.item())

        loss.backward()
        optimizer.step()
    
    return loss_per_batch


def validate(model,val_dataloader, loss_function):

    model.eval() # --> Setting the model to evaluation mode

    loss_per_batch = []

    with torch.no_grad():

        for images,labels in val_dataloader:
            images,labels = images.to(device),labels.to(device)
            output = model(images)
            labels = torch.squeeze(labels)

            loss = loss_function(output,labels)
            loss_per_batch.append(loss.item())
    
    return loss_per_batch


    # # running_loss = 0.0
    # for i, data in enumerate(train_data, 0):
    #     # get the inputs; data is a list of [inputs, labels]
    #     # inputs, labels = data           
    #     images, labels = data[0].to(device), data[1].to(device)
    #     labels = torch.squeeze(labels)
    #     # print('label shape',labels.shape)
    #     # zero the parameter gradients
    #     optimizer.zero_grad()

    #     # forward + backward + optimize
    #     outputs = model(images)
    #     # print('forward pass shape',outputs.shape)
    #     loss = loss_function(outputs, labels)
    #     loss.backward()
    #     optimizer.step()

    #     _,prediction = torch.max(outputs,1)

    #     total_correct += (prediction == labels).sum().item()
    #     total_samples += labels.size(0)


    #     # print statistics
    #     running_loss += loss.item()
    #     if i % 2000 == 1999:    # print every 2000 mini-batches
    #         print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
    #         running_loss = 0.0

    # accuracy = 100 * total_correct / total_samples
    # print(f'Epoch {epoch+1}: Accuracy = {accuracy:.2f}%')

# print('Finished Training')
# torch.save(cnn_model.state_dict(), PATH)

def calculate_accuracy(model,dataloader):

    model.eval() # Turns of Specific Parameters like BatchNorm , Dropout

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images , labels in tqdm(dataloader):
            images,labels = images.to(device),labels.to(device)

            predictions = torch.argmax(model(images),dim=1)

            correct_prediction = sum(predictions==labels).items()

            total_correct += correct_prediction
            total_samples += len(images)

    return round(total_correct/total_samples , 3)


if __name__ == '__main__':


    # wandb.login()

    # run = wandb.init(
    #     project = 'dl_ass2'
    # )

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


    



    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(3):
      print(f'Epoch {epoch+1}/{3}')
      train_losses = []

      #  training
      print('training...')
      for images, labels in tqdm(train_dataloader):
        #  sending data to device
        images, labels = images.to(device), labels.to(device)
        #  resetting gradients
        optimizer.zero_grad()
        #  making predictions
        predictions = cnn_model(images)
        #  computing loss
        labels = torch.squeeze(labels)

        loss = loss_function(predictions, labels)
        # log_dict['training_loss_per_batch'].append(loss.item())
        train_losses.append(loss.item())
        #  computing gradients
        loss.backward()
        #  updating weights
        optimizer.step()
      with torch.no_grad():
        print('deriving training accuracy...')
        #  computing training accuracy
        train_accuracy = calculate_accuracy(cnn_model, train_dataloader)
        print('train_accuracy',train_accuracy)
        # log_dict['training_accuracy_per_epoch'].append(train_accuracy)

      #  validation
      print('validating...')
      val_losses = []

      #  setting convnet to evaluation mode
      cnn_model.eval()

      with torch.no_grad():
        for images, labels in tqdm(val_dataloader):
          #  sending data to device
          images, labels = images.to(device), labels.to(device)
          #  making predictions
          predictions = cnn_model(images)
          #  computing loss
          labels = torch.squeeze(labels)

          val_loss = loss_function(predictions, labels)
        #   log_dict['validation_loss_per_batch'].append(val_loss.item())
          val_losses.append(val_loss.item())
        #  computing accuracy
        print('deriving validation accuracy...')
        val_accuracy = calculate_accuracy(cnn_model, val_dataloader)
        # log_dict['validation_accuracy_per_epoch'].append(val_accuracy)

      train_losses = np.array(train_losses).mean()
      val_losses = np.array(val_losses).mean()

      print(f'training_loss: {round(train_losses, 4)}  training_accuracy: '+
      f'{train_accuracy}  validation_loss: {round(val_losses, 4)} '+  
      f'validation_accuracy: {val_accuracy}\n')



    # train_loss = train(cnn_model,train_dataloader,optimizer,loss_function)
    # val_loss = validate(cnn_model,val_dataloader,optimizer,loss_function)

    # train_accuracy = calculate_accuracy(cnn_model,train_dataloader)
    # val_accuracy = calculate_accuracy(cnn_model,val_dataloader)

    # print('train acc:',train_accuracy," val acc: ",val_accuracy)














### Down might be waste










    # for epoch in tqdm(range(2)):  # loop over the dataset multiple times
    #     total_correct = 0
    #     total_samples = 0
    #     running_loss = 0.0
    #     for i, data in enumerate(train_dataloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         # inputs, labels = data           
    #         images, labels = data[0].to(device), data[1].to(device)
    #         labels = torch.squeeze(labels)
    #         # print('label shape',labels.shape)
    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = cnn_model(images)
    #         # print('forward pass shape',outputs.shape)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         _,prediction = torch.max(outputs,1)

    #         total_correct += (prediction == labels).sum().item()
    #         total_samples += labels.size(0)


    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:    # print every 2000 mini-batches
    #             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
    #             running_loss = 0.0

    #     accuracy = 100 * total_correct / total_samples
    #     print(f'Epoch {epoch+1}: Accuracy = {accuracy:.2f}%')

    # print('Finished Training')
    # torch.save(cnn_model.state_dict(), PATH)


