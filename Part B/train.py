import torchvision.models as models
from torch import nn
from preprocess import Custom_train_dataset,Custom_val_dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from CNN_train import CNN_train






if __name__ == "__main__":

    PRETRAINED_MODEL_NAME = 'vgg11' # Later get from argsparse


    # if PRETRAINED_MODEL_NAME == 'googlenet':
    #     model = models.googlenet(pretrained=True)
    # elif PRETRAINED_MODEL_NAME == 'inceptionv3':
    #     model = models.inception_v3(pretrained=True)
    # elif PRETRAINED_MODEL_NAME == 'resnet50':
    #     model = models.resnet50(pretrained=True)
    # elif PRETRAINED_MODEL_NAME == 'vgg':
    #     model = models.vgg16(pretrained=True)
    # else:
    #     raise ValueError("Invalid model name. Please choose from: googlenet, inceptionv3, resnet50, vgg")
    


    # Changing parameters wrt our use case
    model = models.vgg16(pretrained=True)

    num_feat = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_feat,10)



    train_dataset = Custom_train_dataset(dataset_path = '../Train_Val_Dataset/train')
    val_dataset = Custom_val_dataset(dataset_path = '../Train_Val_Dataset/val')


    train_dataloader = DataLoader(train_dataset, batch_size=64 ,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model.to(device)   
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(),lr=0.0001)
    trainer = CNN_train(model,train_dataloader,val_dataloader,optimizer,loss_function,device)

    trainer.fit()

