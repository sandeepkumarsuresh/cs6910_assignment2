import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import os
import cv2

class Custom_dataset(Dataset):
    # ref: https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d
    # Here we need to override two methods:__len__ and __getitem__

    def __init__(self,dataset_path):
        self.dataset_path = dataset_path
        self.data = []
        for data_class in os.listdir(dataset_path):
            for images in os.listdir(os.path.join(dataset_path,data_class)):
                image_path = os.path.join(dataset_path,data_class,images)
                self.data.append([image_path,data_class])
        print(len(self.data))
        self.img_dims = (415,415)
        self.class_map = {
            "Fungi": 0,
            "Insecta": 1,
            "Animalia": 2,
            "Arachnida": 3,
            "Aves": 4,
            "Mollusca": 5,
            "Reptilia": 6,
            "Plantae": 7,
            "Amphibia": 8,
            "Mammalia": 9
        }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path,class_name = self.data[index]
        img = cv2.imread(img_path)
        img = cv2.resize(img,self.img_dims)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        class_id = torch.tensor([class_id])
        return img_tensor,class_id

if __name__ == '__main__':

    dataset = Custom_dataset(dataset_path='./Datasets/inaturalist_12K/train')
    
