import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import cv2

data_tf = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()])

#准备训练数据
train_set = torchvision.datasets.MNIST('./data', train=True,  transform=data_tf)
batchSize = 64
train_data = DataLoader(train_set, batch_size=batchSize, shuffle=True) # 64
# 在装载完成后，我们可以选取其中一个批次的数据进行预览
images,labels = next(iter(train_data))
img = torchvision.utils.make_grid(images)
 
img = img.numpy().transpose(1,2,0)
print(labels)
cv2.imshow('images',img)
key_pressed=cv2.waitKey(0)