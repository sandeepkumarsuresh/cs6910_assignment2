"""
This script will modify the class of the pretrained model to 
our requirement

https://medium.com/analytics-vidhya/how-to-add-additional-layers-in-a-pre-trained-model-using-pytorch-5627002c75a5
"""

import torch
from torch import nn
import torchvision.models as models

class Custom_VGG_Add_Layer(nn.Module):
    def __init__(self, num_classes):
        super(Custom_VGG_Add_Layer, self).__init__()

        vgg_model = models.vgg16(pretrained = True)

        for feature_extractors in vgg_model.features.parameters():
            feature_extractors.requires_grad = False
        
        feature_in = vgg_model.classifier[6].in_features
        vgg_model.classifier[6] = nn.Linear(feature_in,feature_in)

        FC_layer2 = nn.Sequential(
            nn.Linear(feature_in,num_classes),
            nn.Softmax()
        )
        vgg_model.classifier.add_module("FC_layer2",FC_layer2)

        self.model = vgg_model

    def forward(self,x):
        return self.model(x)



# TODO : Add Multiple Configurations in class Here

class Custom_VGG_Modify_LastLayer(nn.Module):

    def __init__(self, num_classes):
        super(Custom_VGG_Modify_LastLayer, self).__init__()

        vgg_model = models.vgg16(pretrained = True)

        for feature_extractors in vgg_model.features.parameters():
            feature_extractors.requires_grad = False
        
        feature_in = vgg_model.classifier[6].in_features
        vgg_model.classifier[6] = nn.Linear(feature_in,num_classes)

        # FC_layer2 = nn.Sequential(
        #     nn.Linear(feature_in,num_classes),
        #     nn.Softmax()
        # )
        # vgg_model.classifier.add_module("FC_layer2",FC_layer2)

        self.model = vgg_model

    def forward(self,x):
        return self.model(x)
    


class Custom_VGG_Freezing_Layers(nn.Module):

    def __init__(self, num_classes,layers_to_freeze):
        super(Custom_VGG_Freezing_Layers, self).__init__()

        vgg_model = models.vgg16(pretrained = True)

        num_layers = len(list(vgg_model.features.children()))
        for i, layer in enumerate(vgg_model.features.children()):
            if i >= num_layers - layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False
 
        feature_in = vgg_model.classifier[6].in_features
        vgg_model.classifier[6] = nn.Linear(feature_in,num_classes)


        # for feature_extractors in vgg_model.features.parameters():
        #     feature_extractors.requires_grad = False
        
        # feature_in = vgg_model.classifier[6].in_features
        # vgg_model.classifier[6] = nn.Linear(feature_in,num_classes)

        # FC_layer2 = nn.Sequential(
        #     nn.Linear(feature_in,num_classes),
        #     nn.Softmax()
        # )
        # vgg_model.classifier.add_module("FC_layer2",FC_layer2)

        self.model = vgg_model

    def forward(self,x):
        return self.model(x)
    
