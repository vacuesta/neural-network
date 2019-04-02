from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


ModifyPics=transforms.Compose([transforms.Resize(100),
                                    transforms.CenterCrop(100),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])


# Make your dataset ready for the network

train_set =ImageFolder("Dog-cat-data/training_set/",transform=ModifyPics)
train_loader=torch.utils.data.DataLoader(train_set,batch_size=50, shuffle=True)

test_set=ImageFolder("Dog-cat-data/test_set/",transform=ModifyPics)
test_loader=torch.utils.data.DataLoader(test_set,batch_size=50,shuffle=True)

