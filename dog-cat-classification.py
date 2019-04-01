from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
input_size = 1000
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001



class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

ModifyPics=transforms.Compose([transforms.Resize(100),
                                    transforms.CenterCrop(100),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])


# Make your dataset ready for the network

train_set =ImageFolder("Dog-cat-data/training_set/",transform=ModifyPics)
train_loader=torch.utils.data.DataLoader(train_set,batch_size=50, shuffle=True)

test_set=ImageFolder("Dog-cat-data/test_set/",transform=ModifyPics)
test_loader=torch.utils.data.DataLoader(test_set,batch_size=50,shuffle=True)

