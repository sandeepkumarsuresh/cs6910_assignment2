"""
https://discuss.huggingface.co/t/do-you-train-all-layers-when-fine-tuning-t5/1034/8
"""

import torchvision.models as models
from torch import nn
from preprocess import Custom_train_dataset,Custom_val_dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from CNN_train_partB import CNN_train
from pretrained_model import Custom_VGG_Add_Layer,Custom_VGG_Freezing_Layers,Custom_VGG_Modify_LastLayer
import wandb
import os


if __name__ == "__main__":

    PRETRAINED_MODEL_NAME = 'vgg16' # Later get from argsparse

    MODEL_SAV_DIR = "./model_dir"
    os.makedirs(MODEL_SAV_DIR,exist_ok=True)


    # model = models.vgg16(pretrained=True)
    # model = Custom_VGG_Add_Layer(num_classes=10)
    model = Custom_VGG_Modify_LastLayer(num_classes=10)
    # model = Custom_VGG_Freezing_Layers(num_classes=10,layers_to_freeze=3)

    print(model)
    train_dataset = Custom_train_dataset(dataset_path = '../Train_Val_Dataset/train')
    val_dataset = Custom_val_dataset(dataset_path = '../Train_Val_Dataset/val')


    train_dataloader = DataLoader(train_dataset, batch_size=64 ,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)




    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


    model.to(device)   
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    trainer = CNN_train(model,train_dataloader,val_dataloader,optimizer,loss_function,device)

    trainer.fit()

    torch.save(model.state_dict())
