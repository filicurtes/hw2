import os
import logging
import os.path
import sys
import re
import cv2
import numpy as np
import pandas as pd 
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
import torchvision.models as models
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
from plotting import plot



DEVICE = 'cuda' # 'cuda' or 'cpu'

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

set='prova'

NUM_CLASSES = 101 # 101 + 1: There is am extra Background class that should be removed 

BATCH_SIZE = 161
     # Higher batch sizes allows for larger learning rates. An empirical heuristic suggests that, when changing
                     # the batch size, learning rate should change by the same factor to have comparable results

LR = 1e-3           # The initial Learning Rate
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default

NUM_EPOCHS = 5     # Total number of training epochs (iterations over dataset)
STEP_SIZE = 20      # How many epochs before decreasing learning rate (if using a step-down policy)
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

val_dataset = Subset(train_dataset, val_indexes)
train_dataset = Subset(train_dataset, train_indexes)

# Check dataset sizes
print('Train Dataset: {}'.format(len(train_dataset)))
print('Valid Dataset: {}'.format(len(val_dataset)))
print('Test Dataset: {}'.format(len(test_dataset)))

train_classes = np.zeros(101)

for elem in train_dataset:
  train_classes[elem[1]] += 1

val_classes= np.zeros(101)

for elem in val_dataset:
  val_classes[elem[1]] += 1

print(train_classes)
ax=sns.barplot(x=np.linspace(0, 100, num=101),y=train_classes)
plt.savefig(f'{set}myfig.png')


# Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(type(train_dataloader))

net = models.vgg16_bn(pretrained=True) # Loading AlexNet model

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
optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,nesterov=True)

# Define scheduler
# A scheduler dynamically changes learning rate
# The most common schedule is the step(-down), which multiplies learning rate by gamma every STEP_SIZE epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# By default, everything is loaded to cpu
net = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda

cudnn.benchmark # Calling this optimizes runtime

current_step = 0

loss_train_list=[]

acc_train_list=[]

acc_val_list=[]

loss_val_list=[]


def validation(net):
  ##VALIDATION
  net.train(False)
  running_corrects_val = 0
  loss_epoch_list_val=[]
  with torch.no_grad():
  
    for images,labels in val_dataloader:
      #Bring data over the device of choice
      images=images.to(DEVICE)
      labels=labels.to(DEVICE)

      # Forward Pass
      outputs = net(images)

      ##Loss_validation
      loss_val = criterion(outputs, labels)
      
      
      loss_epoch_list_val.append(loss_val)
      # Get predictions
      _, preds = torch.max(outputs.data, 1)

      # Update Corrects
      running_corrects_val += torch.sum(preds == labels.data).data.item()

    # Calculate Accuracy
    accuracy_val = running_corrects_val / float(len(val_dataset))

    print('Validation Accuracy: {}'.format(accuracy_val))
    acc_val_list.append(accuracy_val)

    loss_avg=sum(loss_epoch_list_val)/len(loss_epoch_list_val)
    loss_nump=loss_avg.data.cpu().numpy()
    loss_val_list.append(loss_nump)
  
def training(net,current_step):

  running_corrects_train=0

  loss_epoch_list=[]

  # Iterate over the dataset
  for images, labels in train_dataloader:
    # Bring data over the device of choice
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    net.train() # Sets module in training mode

    # PyTorch, by default, accumulates gradients after each backward pass
    # We need to manually set the gradients to zero before starting a new iteration
    optimizer.zero_grad() # Zero-ing the gradients

    # Forward pass to the network
    outputs = net(images)

    # Get predictions
    _, preds = torch.max(outputs.data, 1)

    # Update Corrects
    running_corrects_train += torch.sum(preds == labels.data).data.item()

    # Compute loss based on output and ground truth
    loss = criterion(outputs, labels)
    
    loss_epoch_list.append(loss)

    # Log loss
    if current_step % LOG_FREQUENCY == 0:
      print('Step {}, Loss {}'.format(current_step, loss.item()))
    
    # Compute gradients for each layer and update weights
    loss.backward()  # backward pass: computes gradients
    

    optimizer.step() # update weights based on accumulated gradients


    current_step += 1
    
  # Calculate Accuracy
  accuracy_train = running_corrects_train / float(len(train_dataset))
  acc_train_list.append(accuracy_train)

  # Step the scheduler
  scheduler.step() 

  


  loss_avg=sum(loss_epoch_list)/len(loss_epoch_list)

  loss_nump=loss_avg.data.cpu().numpy()
  loss_train_list.append(loss_nump)
  print(loss_nump)


# Start iterating over the epochs
for epoch in range(NUM_EPOCHS):
  print('Starting epoch {}/{}, LR = {}'.format(epoch+1, NUM_EPOCHS, scheduler.get_lr()))
  
  training(net,current_step)
  validation(net)

plot(loss_train_list,loss_val_list,'loss','loss_train','loss_val','Loss vs Epochs',f'VGG_loss_train_pre',set)

plot(acc_train_list,acc_val_list,'accuracy','accuracy_train','accuracy_val','Accuracy vs Epochs',f'VGG_acc_train_pre',set)


owd=os.getcwd()
os.chdir('files')
file1 = open(f'{set}.txt',"w+") 
file1.write(f'BATCH_SIZE:{BATCH_SIZE}\nLR:{LR}\nMOMENTUM:{MOMENTUM}\nWEIGHT_DECAY:{WEIGHT_DECAY}\nNUM_EPOCHS:{NUM_EPOCHS}\nSTEP_SIZE:{STEP_SIZE}\nGAMMA:{GAMMA}\nLOG_FREQUENCY:{LOG_FREQUENCY}')


loss_train_arr=np.array(loss_train_list)
acc_val_arr=np.array(acc_val_list)

results={'acc_train':acc_train_list,'loss_train':loss_train_arr,'acc_val':acc_val_arr,'loss_val':loss_val_list}
results_df=pd.DataFrame(results)
results_df.to_csv(f'{set}_VGG_results.csv',index=False)
os.chdir(owd)