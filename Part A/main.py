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
from preprocess import Custom_train_dataset , Custom_val_dataset
from CNN import CNN
import wandb
from CNN_train import CNN_train



sweep_configuration = {
    'method': 'random', #grid, random
    'metric': {
    'name': 'validation_accuracy',
    'goal': 'maximize'   
    },
    'parameters': {
        'n_filters': {
            'values': [64,128,256]
        },
        'data_augmentation': {
            'values': ['True']
        },
        'batch_normalisation': {
            'values': ['True']
        },
        'dropout': {
            'values': [0.2,0.3]
        },
        'batch_size':{
            'values': [32]
        },
        'filter_multiplier':{
            'values':[0.5,1,2]
        },
        'kernel_size':{
            'values':[3,5]
        }
    }
}
sweep_id = wandb.sweep(sweep_configuration,project='dl_ass2')


def do_sweep():

    wandb.init()
    config = wandb.config
    run_name = "Adam Optimizer:"+"batch_size:"+str(config.batch_size)+"filters:"+str(config.n_filters) + "dropout:"+str(config.dropout)+"filter_multiplier:"+str(config.filter_multiplier)+"kernel_size:"+str(config.kernel_size)
    print(run_name)
    wandb.run.name = run_name

    train_dataset = Custom_train_dataset(dataset_path = '../Train_Val_Dataset/train')
    val_dataset = Custom_val_dataset(dataset_path = '../Train_Val_Dataset/val')


    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size ,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cnn_model = CNN(

        num_filter=config.n_filters,
        filter_multiplier=config.filter_multiplier,
        dropout=config.dropout,
        kernel_size=config.kernel_size
        
    )

    cnn_model.to(device)   

    print(cnn_model)

    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)

    # optimizer = optim.Adam(cnn_model.parameters(), lr=0.0001)

    optimizer = optim.NAdam(cnn_model.parameters())




    trainer = CNN_train(cnn_model,train_dataloader,val_dataloader,optimizer,loss_function,device)

    trainer.fit()

    torch.save(cnn_model.state_dict(), f'{MODEL_SAV_DIR}/{run_name}')


if __name__ == '__main__':


    # wandb.login()

    # run = wandb.init(
    #     project = 'dl_ass2'
    # )

    # PATH = './model.pth'
    MODEL_SAV_DIR = "./model_dir"
    os.makedirs(MODEL_SAV_DIR,exist_ok=True)


    # splitfolders.ratio("Datasets/inaturalist_12K/train", output="Train_Val_Dataset",
    #                     seed=1337, ratio=(.8, .2), group_prefix=None, move=False)
    

    wandb.agent(sweep_id ,function=do_sweep,count=100)
    wandb.finish()

    # train_dataset = Custom_dataset(dataset_path = 'Train_Val_Dataset/train')
    # val_dataset = Custom_dataset(dataset_path = 'Train_Val_Dataset/val')


    # train_dataloader = DataLoader(train_dataset, batch_size=32 ,shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print("Device Available : ",device)

    # cnn_model = CNN()

    # cnn_model.to(device)


    



    # loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)


    # trainer = CNN_train(cnn_model,train_dataloader,val_dataloader,optimizer,loss_function,device)

    # trainer.fit()
