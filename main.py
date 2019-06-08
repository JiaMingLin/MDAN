#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import constant
from utils import get_logger, get_lr
from datasets import data_loader as dl

import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import shutil

from tensorboardX import SummaryWriter
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--target", help="The target domain", type=str)
parser.add_argument("-n", "--name", help="Test case display name.", type=str)
parser.add_argument("-e", "--total_epoch", help="Number of training epochs", type=int, default=3)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=4)
parser.add_argument("-c", "--class_number", help="The number of class.", type=int, default=0)
parser.add_argument("-r", "--learning_rate", help="Learning rate during training", type=float, default=0.001)
parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
"""
parser.add_argument("-f", "--frac", help="Fraction of the supervised training data to be used.",
                    type=float, default=1.0)

parser.add_argument("-v", "--verbose", help="Verbose mode: True -- show training progress. False -- "
                                            "not show training progress.", type=bool, default=True)
parser.add_argument("-m", "--model", help="Choose a model to train: [mdan]",
                    type=str, default="mdan")
# The experimental setting of using 5000 dimensions of features is according to the papers in the literature.
parser.add_argument("-d", "--dimension", help="Number of features to be used in the experiment",
                    type=int, default=5000)
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                    type=float, default=1e-2)
parser.add_argument("-o", "--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str, default="dynamic")
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## ===============================
# Compile and configure all the model parameters.
## ===============================
args = parser.parse_args()
target = args.target
name = args.name
total_epoch = args.total_epoch
batch_size = args.batch_size
class_number = args.class_number
learning_rate = args.learning_rate
seed = args.seed


test_case_place = os.path.join(constant.logs_root, name)
if os.path.isdir(test_case_place):
    shutil.rmtree(test_case_place)
os.makedirs(test_case_place)
    
logger = get_logger(os.path.join(test_case_place, 'messages'))  # amazon
writer = SummaryWriter(
            log_dir = os.path.join(test_case_place, 'eval')
        )

# Set random number seed.
np.random.seed(seed)
torch.manual_seed(seed)

## ==============================
# Loading Dataset
## ==============================
# Loading the randomly partition the amazon data set.
#image_size = (512,512)
image_size = (224,224)
train_val_batch_size = {
    'train':batch_size,
    'val':1
}
image_dataset = {
    'train': dl.load_data(target, is_train = True, resize = image_size, class_num = class_number),
    'val': dl.load_data(target, is_train = False, resize = image_size, class_num = class_number)
}

dataloaders = {
    'train': DataLoader(image_dataset['train'], batch_size=train_val_batch_size['train'],shuffle=True, num_workers=8),
    'val': DataLoader(image_dataset['val'], batch_size=train_val_batch_size['val'],shuffle=False, num_workers=8),
}

dataset_sizes = {
    'train': len(image_dataset['train']),
    'val': len(image_dataset['val'])
}
class_names = image_dataset['train'].classes

logger.info('training set size = {}'.format(dataset_sizes['train']))
logger.info('testing set size = {}'.format(dataset_sizes['val']))
logger.info("-" * 100)


## ==============================
# Setup Model
## ==============================
model = models.alexnet(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
#num_ftrs = model.classifier.in_features
classifier = nn.Sequential(
    nn.Dropout(),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, len(class_names)),
)
model.classifier = classifier
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

## ==============================
# Training Model
## ==============================
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0
train_loss = []
train_acc = []

val_loss = []
val_acc = []

logger.info('Starting to train...')
def train(epoch):
    scheduler.step()
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    iteration = 0
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()    
        
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        iteration += 1
        if iteration % 500 == 0:
            logger.info('Epoch {}/{}, Iteration {}/{}, Training Loss {:.4f}, Training Accuracy {:.4f}'.format(
                epoch, total_epoch,
                iteration, int(dataset_sizes['train']/float(train_val_batch_size['train'])),
                (running_loss/(float(train_val_batch_size['train']) * iteration)), 
                (running_corrects/(float(train_val_batch_size['train']) * iteration)) 
            ))
        
        
    epoch_loss = running_loss / dataset_sizes['train']
    epoch_acc = running_corrects.double() / dataset_sizes['train']
    logger.info('Epoch {}/{}, Training Loss: {:.4f}, Training Acc: {:.4f}'.format(
                epoch, total_epoch,
                epoch_loss, epoch_acc))
    # write to tensorborad
    writer.add_scalar('Training_Loss', epoch_loss)
    writer.add_scalar('Training_Accuracy', epoch_acc)
    writer.add_scalar('Learning_Rate', get_lr(optimizer))
    
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)

def val(epoch):
    logger.info('Validating the model...')
    model.eval()   # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    # Iterate over data.
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes['val']
    epoch_acc = running_corrects.double() / dataset_sizes['val']
            
    logger.info('Epoch {}/{}, Validation Loss: {:.4f}, Validation Acc: {:.4f}'.format(
                epoch, total_epoch,
                epoch_loss, epoch_acc))
    # write to tensorborad
    writer.add_scalar('Validation_Loss', epoch_loss)
    writer.add_scalar('Validation_Accuracy', epoch_acc)
    
    # deep copy the model
    global best_acc
    if epoch_acc >= best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': {'train': train_loss, 'val': val_loss},
            'acc': {'train': train_acc, 'val': val_acc}
            }, os.path.join(test_case_place, 'best_model.pt')
        )
        
    val_loss.append(epoch_loss)
    val_acc.append(epoch_acc)

for epoch in range(total_epoch):
    train(epoch)
    #val(epoch)
    if (epoch+1) % 5 == 0:
        val(epoch)
## ==============================
# Produce Submit
## ==============================
test_dataset = dl.load_data('test', resize = image_size)
test_dataloader = DataLoader(test_dataset, batch_size=1,shuffle=False, num_workers=8)

model.load_state_dict(best_model_wts)
predictions = []

logger.info('Predicting the test images...')
for inputs in test_dataloader:
    inputs = inputs.to(device)

    # forward
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predictions.append(preds)
        
predictions = [data.item() for data in predictions]
result = []
for (image_path, pred) in  zip(test_dataset.img_paths, predictions):
    result.append('{},{}\n'.format(image_path, pred))
    
with open(os.path.join(test_case_place, 'submit_{}.txt'.format(name)), 'w+') as fout:
    fout.write('image_name,label\n')
    for line in result:
        fout.write(line)
        
logger.info('Done!')
