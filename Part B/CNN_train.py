import torch
import numpy as np
from tqdm import tqdm
import wandb
class CNN_train():

    def __init__(self,model,train_dataloader,val_dataloader,optimizer,loss_function,device,epoch = 10):
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.val_dataloader = val_dataloader
        self.epoch = epoch

    def train_cnn(self):

        self.model.train() # Tells the self.model that you are doing the training

        loss_per_batch = []
        for images,labels in self.train_dataloader:

            images, labels = images.to(self.device),labels.to(self.device)
            self.optimizer.zero_grad()
            
            output = self.model(images)
            labels = torch.squeeze(labels)
            loss = self.loss_function(output,labels)
            loss_per_batch.append(loss.item())

            loss.backward()
            self.optimizer.step()
        
        return loss_per_batch


    def validate_cnn(self):

        self.model.eval() # --> Setting the self.model to evaluation mode

        loss_per_batch = []

        with torch.no_grad():

            for images,labels in self.val_dataloader:
                images,labels = images.to(self.device),labels.to(self.device)
                output = self.model(images)
                labels = torch.squeeze(labels)

                loss = self.loss_function(output,labels)
                loss_per_batch.append(loss.item())
        
        return loss_per_batch


    def calculate_accuracy(self,dataloader):

        self.model.eval() # Turns of Specific Parameters like BatchNorm , Dropout

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images , labels in tqdm(dataloader):
                images,labels = images.to(self.device),labels.to(self.device)

                predictions = torch.argmax(self.model(images),dim=1)
                labels = torch.squeeze(labels)

                correct_prediction = sum(predictions==labels).item()

                total_correct += correct_prediction
                total_samples += len(images)

        return round(total_correct/total_samples , 3)

    def fit(self):
    
        for epoch in range(self.epoch):
            print(f'Epoch {epoch+1}/{self.epoch}')
            train_losses = []

            train_losses.append(self.train_cnn())


            train_accuracy = self.calculate_accuracy(self.train_dataloader)
            print('train_accuracy',train_accuracy)

            val_losses = []

            val_losses.append(self.validate_cnn())

            val_accuracy = self.calculate_accuracy(self.val_dataloader)

            print('validation_accuracy',val_accuracy)
                # log_dict['validation_accuracy_per_epoch'].append(val_accuracy)

            train_losses = np.array(train_losses).mean()
            val_losses = np.array(val_losses).mean()
            # wandb.log({'training_loss':round(train_losses, 4) , "training_accuracy": train_accuracy ,  "validation_loss": round(val_losses, 4)  , "validation_accuracy":val_accuracy })
            print(f'training_loss: {round(train_losses, 4)}  training_accuracy: '+f'{train_accuracy}  validation_loss: {round(val_losses, 4)} '+ f'validation_accuracy: {val_accuracy}\n')