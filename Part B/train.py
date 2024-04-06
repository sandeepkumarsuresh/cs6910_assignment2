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


sweep_configuration = {
    'method': 'bayes', #grid, random
    'metric': {
    'name': 'validation_accuracy',
    'goal': 'maximize'   
    },
    'parameters': {
        'strategy': {
            'values': ['Add_layer','Modify_Last_Layer','Freezing_Layers']
        },
        'pretrained_models': {
            'values': ['vgg16','vgg16_bn','vgg19','vgg19_bn']
        },
        'dropout': {
            'values': [0,0.2,0.3]
        },
        'batch_size':{
            'values': [32,64]
        }
    }
}
sweep_id = wandb.sweep(sweep_configuration,project='dl_ass2')

def do_sweep():

    wandb.init()
    config = wandb.config
    run_name = "Part B "+"strategy"+str(config.strategy)+"pretrained model "+str(config.pretrained_models)+"batch_size:"+str(config.batch_size) + "dropout:"+str(config.dropout)
    print(run_name)
    wandb.run.name = run_name

    if config.strategy == 'Add_layer':
        model = Custom_VGG_Add_Layer(config.pretrained_models,num_classes=10,dropout=config.dropout)
    elif config.strategy == 'Modify_Last_Layer':
        model = Custom_VGG_Modify_LastLayer(config.pretrained_models,num_classes=10)
    elif config.strategy == 'Freezing_Layers':
        model = Custom_VGG_Freezing_Layers(config.pretrained_models,num_classes=10,layers_to_freeze=3)
    
    train_dataset = Custom_train_dataset(dataset_path = '../Train_Val_Dataset/train')
    val_dataset = Custom_val_dataset(dataset_path = '../Train_Val_Dataset/val')

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size ,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)    

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


    model.to(device)   
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    trainer = CNN_train(model,train_dataloader,val_dataloader,optimizer,loss_function,device,epoch=3)

    trainer.fit()
    
    file_path = f'{MODEL_SAV_DIR}/{config.strategy}_{config.pretrained_models}_{config.dropout}_{config.batch_size}'
    
    torch.save(model.state_dict(),file_path)


if __name__ == "__main__":

    PRETRAINED_MODEL_NAME = 'vgg16' # Later get from argsparse

    MODEL_SAV_DIR = "./model_dir"
    os.makedirs(MODEL_SAV_DIR,exist_ok=True)

    wandb.agent(sweep_id ,function=do_sweep,count=100)
    wandb.finish()

