%matplotlib inline
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms

#Because we are running this program with a CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#training ser of 60,000
train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)
#print(len(train_dataset))

#test dataser of 10,000
test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transforms.ToTensor(),download = True)

#600
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

#100
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=100, shuffle=False)

#Displayes our first 20 digits
images, labels = next(iter(test_loader))
for i in range(20):
    x=images[i]
    x=np.array(x, dtype='float')
    pixels = x.reshape((28, 28))
   #plt.imshow(pixels, cmap='gray')
   # plt.show()

    
    
#Building Our neural Network
class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network,self).__init__()
        self.layer_one = nn.Linear(784,100)
        self.layer_two = nn.Linear(100,50)
        self.fin_layer = nn.Linear(50, 10) #Output is 10 classes
    
    def forward(self, x):
        x = x.view(-1, 784) #Changed size to vector
        x = F.relu(self.layer_one(x))
        x = F.relu(self.layer_two(x))
        x = self.fin_layer(x)
        return x
        
       
        
        

model = Neural_Network() #Our class
criterion = nn.NLLLoss() #our cost function. there are many more but this one is good for multiple classes
optimizer = optim.Adam(model.parameters(), lr=.05) #our optimizer object for our weights
epochs = 5

model.to(device)

#Training
for i in range(epochs):
    running_loss=0
    for images, target in train_loader:
        images,target = images.to(device), target.to(device)
        
        inputs = model(images)
        loss = criterion(inputs, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("output {}".format(running_loss))
        
        
        
        


    
        
        
        
