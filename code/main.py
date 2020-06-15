import os
import logging
import os.path
import sys
import re
import cv2
import numpy as np
import time

from collections import defaultdict

import matplotlib.pyplot as plt

import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader,Dataset
from torch.backends import cudnn
from torchvision.datasets import VisionDataset
from torch.utils.data.sampler import SubsetRandomSampler
import pretrainedmodels
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from torchvision.models import alexnet
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


DEVICE = 'cuda' # 'cuda' or 'cpu'

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()



NUM_CLASSES = 101 # 101 + 1: There is am extra Background class that should be removed 

BATCH_SIZE = 256     # Higher batch sizes allows for larger learning rates. An empirical heuristic suggests that, when changing
                     # the batch size, learning rate should change by the same factor to have comparable results

LR = 1e-1            # The initial Learning Rate
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default

NUM_EPOCHS = 20     # Total number of training epochs (iterations over dataset)
STEP_SIZE = 2      # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1          # Multiplicative factor for learning rate step-down

LOG_FREQUENCY = 10

# Define transforms for training phase
train_transform = transforms.Compose([transforms.Resize(256),      # Resizes short size of the PIL image to 256
                                      transforms.CenterCrop(224),  # Crops a central square patch of the image
                                                                   # 224 because torchvision's AlexNet needs a 224x224 input!
                                                                   # Remember this when applying different transformations, otherwise you get an error
                                      transforms.ToTensor(),       # Turn PIL Image to torch.Tensor
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizes tensor with mean and standard deviation
])
# Define transforms for the evaluation phase
eval_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                    
])

DATA_DIR = '101_ObjectCategories'
from caltech_dataset import Caltech

# Prepare Pytorch train/test Datasets
train_dataset = Caltech(DATA_DIR, split='train',  transform=train_transform)
test_dataset = Caltech(DATA_DIR, split='test', transform=eval_transform)
print(type(train_dataset))

# split to train val
train_len = int(train_dataset.__len__() * 0.5)
val_len = int(train_dataset.__len__() * 0.5)
train_indexes = np.arange(train_dataset.__len__())
print(train_indexes)
train_labels = np.empty(train_dataset.__len__(), dtype=int)


for index in train_indexes:
  train_labels[index] = train_dataset.__getitem__(index)[1]
print(train_labels)  

train_indexes, val_indexes, _, _ = train_test_split(train_indexes, train_labels, test_size=0.5, random_state=42, stratify=train_labels)


"""
validation_split = 0.5
dataset_size = len(train_dataset)
#print(dataset_size)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if 1 :
    np.random.seed(1337)
    np.random.shuffle(indices)
train_indexes, val_indexes = indices[split:], indices[:split]

print(type(train_indexes))
#train_indexes = SubsetRandomSampler(train_indices)
#val_indexes = SubsetRandomSampler(valid_indices)
#train_indexes, val_indexes = 
#print(type(train_indexes))
#print(train_indexes)
"""

val_dataset = Subset(train_dataset, val_indexes)
train_dataset = Subset(train_dataset, train_indexes)

# Check dataset sizes
print('Train Dataset: {}'.format(len(train_dataset)))
print('Valid Dataset: {}'.format(len(val_dataset)))
print('Test Dataset: {}'.format(len(test_dataset)))

train_classes = np.zeros(101)

for elem in train_dataset:
  train_classes[elem[1]] += 1

val_classes = np.zeros(101)

for elem in val_dataset:
  val_classes[elem[1]] += 1

print(train_classes)
ax=sns.barplot(x=np.linspace(0, 100, num=101),y=train_classes)
plt.savefig("myfig.png")


# Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(type(train_dataloader))



net = alexnet() # Loading AlexNet model

# AlexNet has 1000 output neurons, corresponding to the 1000 ImageNet's classes
# We need 101 outputs for Caltech-101
net.classifier[6] = nn.Linear(4096, NUM_CLASSES) # nn.Linear in pytorch is a fully connected layer
                                                 # The convolutional layer is nn.Conv2d

# We just changed the last layer of AlexNet with a new fully connected layer with 101 outputs
# It is strongly suggested to study torchvision.models.alexnet source code

# Define loss function
criterion = nn.CrossEntropyLoss() # for classification, we use Cross Entropy

# Choose parameters to optimize
# To access a different set of parameters, you have to access submodules of AlexNet
# (nn.Module objects, like AlexNet, implement the Composite Pattern)
# e.g.: parameters of the fully connected layers: net.classifier.parameters()
# e.g.: parameters of the convolutional layers: look at alexnet's source code ;) 
parameters_to_optimize = net.parameters() # In this case we optimize over all the parameters of AlexNet

# Define optimizer
# An optimizer updates the weights based on loss
# We use SGD with momentum
optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# Define scheduler
# A scheduler dynamically changes learning rate
# The most common schedule is the step(-down), which multiplies learning rate by gamma every STEP_SIZE epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)


def validate (model,dataloader):
  print('Validating')
  model = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
  model.train(False) # Set Network to evaluation mode
  
  
  running_corrects = 0
  for epoch in range(NUM_EPOCHS):

    for images, labels in tqdm(dataloader):
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

  # Forward Pass
      outputs = model(images)
  # Get predictions
      _, preds = torch.max(outputs.data, 1)

  # Update Corrects
      running_corrects += torch.sum(preds == labels.data).data.item()
    

# Calculate Accuracy
    accuracy = running_corrects / float(len(dataloader.dataset))
      
    print('Validation Accuracy: {}'.format(accuracy))

    
    return accuracy


# training function
def fit(model, dataloader):
    print('Training')
    model = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
    model.train(True)
    train_running_loss = 0.0
    cudnn.benchmark # Calling this optimizes runtime

    current_step = 0
    
    
  # Iterate over the dataset
    for images, labels in dataloader:
    # Bring data over the device of choice
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

      model.train() # Sets module in training mode

    # PyTorch, by default, accumulates gradients after each backward pass
    # We need to manually set the gradients to zero before starting a new iteration
      optimizer.zero_grad() # Zero-ing the gradients

    # Forward pass to the network
      outputs = model(images)

    # Compute loss based on output and ground truth
      loss = criterion(outputs, labels)
      
      train_running_loss += loss.item()
      
    # Log loss
      if current_step % LOG_FREQUENCY == 0:
        print('Step {}, Loss {}'.format(current_step, train_running_loss))
        

    # Compute gradients for each layer and update weights
      loss.backward()  # backward pass: computes gradients
      optimizer.step() # update weights based on accumulated gradients

      current_step += 1

      
      
    
      return train_running_loss

  # Step the scheduler
    scheduler.step()

train_loss=[]
val_accuracy=[]

for epoch in range(NUM_EPOCHS):
      print(f"Epoch {epoch+1} of {NUM_EPOCHS}")   
      trainloss=fit(net, train_dataloader)
      valacc=validate(net, val_dataloader)
      train_loss.append(trainloss)
      val_accuracy.append(valacc)

plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='purple', label='train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

def test (model,dataloader):
  print('Testing')
  model = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
  model.train(False) # Set Network to evaluation mode
  
  
  running_corrects = 0
  for epoch in range(NUM_EPOCHS):

    for images, labels in tqdm(dataloader):
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

  # Forward Pass
      outputs = net(images)
  # Get predictions
      _, preds = torch.max(outputs.data, 1)

  # Update Corrects
      running_corrects += torch.sum(preds == labels.data).data.item()
    

# Calculate Accuracy
    accuracy = running_corrects / float(len(dataloader.dataset))
      
    print('Testing Accuracy: {}'.format(accuracy))

    
    return accuracy

train_loss=[]
test_accuracy=[]

for epoch in range(NUM_EPOCHS):
      print(f"Epoch {epoch+1} of {NUM_EPOCHS}")   
      trainloss=fit(net, train_dataloader)
      testacc=test(net, test_dataloader)
      train_loss.append(trainloss)
      test_accuracy.append(testacc)



plt.figure(figsize=(10, 7))
plt.plot(test_accuracy, color='green', label='test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

