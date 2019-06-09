#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import constant
from utils import get_logger, get_lr
from datasets import data_loader as dl
from models.model_fectory import MDANet

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Test case display name.", type=str, default = 'test_alexnet_multi_1')
parser.add_argument("-e", "--total_epoch", help="Number of training epochs", type=int, default=100)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=32)
parser.add_argument("-c", "--class_number", help="The number of class.", type=int, default=345)
parser.add_argument("-r", "--learning_rate", help="Learning rate during training", type=float, default=0.001)
parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)

## ===============================
# Compile and configure all the model parameters.
## ===============================
args = parser.parse_args()
name = args.name
total_epoch = args.total_epoch
batch_size = args.batch_size
class_number = args.class_number
learning_rate = args.learning_rate
seed = args.seed
gamma = 10
mu = 1e-2
val_model_epoch = 5
train_msg_iter = 10
resize = (224,224)

## ===============================
# Saved result place
## ===============================
test_case_place = os.path.join(constant.logs_root, name)
if os.path.isdir(test_case_place):
    shutil.rmtree(test_case_place)
os.makedirs(test_case_place)

writer = SummaryWriter(
            log_dir = os.path.join(test_case_place, 'eval')
        )

train_dataset_size = {
    'rel': len(dl.load_data('rel', is_train = True, resize = resize, class_num = class_number)),
    'skt': len(dl.load_data('skt', is_train = True, resize = resize, class_num = class_number)),
    'qdr': len(dl.load_data('qdr', is_train = True, resize = resize, class_num = class_number)),
    'inf': len(dl.load_data('inf', is_train = True, resize = resize, class_num = class_number))
}

max_train_dataset_size = max(train_dataset_size.values())
train_dataset_size = sum(train_dataset_size.values())

validate_datasets = {
    'rel': dl.load_data('rel', is_train = False, resize = resize, class_num = class_number),
    'skt': dl.load_data('skt', is_train = False, resize = resize, class_num = class_number),
    'qdr': dl.load_data('qdr', is_train = False, resize = resize, class_num = class_number),
    'inf': dl.load_data('inf', is_train = False, resize = resize, class_num = class_number)
}

validate_dataloader = {
    'rel': DataLoader(validate_datasets['rel'], batch_size=batch_size,shuffle=False, num_workers=8),
    'skt': DataLoader(validate_datasets['skt'], batch_size=batch_size,shuffle=False, num_workers=8),
    'qdr': DataLoader(validate_datasets['qdr'], batch_size=batch_size,shuffle=False, num_workers=8),
    'inf': DataLoader(validate_datasets['inf'], batch_size=batch_size,shuffle=False, num_workers=8)
}

datasets = ['rel', 'skt', 'qdr', 'inf']
#for target in datasets:
for target in ['rel']:
    sources = list(filter(lambda e: e != target, datasets))
    print("Selected sources: ", sources)
    print("Selected target: ", target)
    print("="*50)
    ## ==========================
    # Initialize MDAN model
    ## ==========================
    mdan = MDANet(class_number, len(sources)).to(device)
    optimizer = optim.Adadelta(mdan.parameters(), lr=learning_rate)
    mdan.train()
    
    ## ==========================
    #  MDAN Training
    ## ==========================
    max_iter = int(max_train_dataset_size/batch_size) + 1
    best_acc = 0
    train_loss = []
    val_acc = []
    val_loss = []
    print("Total Epochs: {}, Total iteration: {}/epoch".format(total_epoch, max_iter))
    print("Starting training...")
    
    for epoch in range(total_epoch):
        running_loss = 0.0
        running_cls_loss = 0.0
        running_domain_loss = 0.0
        ## ==========================
        # Sub-Sampling data with replacement = True
        ## ==========================
        multi_data_loader = dl.multiple_data_loader(
            datasets, batch_size, is_train = True, resize = resize, class_num = class_number)
        
        iter_cnt = 0
        
        for batch in multi_data_loader:
            batch_start_time = time.time()
            source_labels = torch.ones(batch_size, requires_grad=False).type(torch.LongTensor).to(device)
            target_labels = torch.zeros(batch_size, requires_grad=False).type(torch.LongTensor).to(device)
            # sub-sampled data from each source and target
            target_batch_img, target_batch_class = batch[target]
            sources_batch = [batch[code] for code in sources]  # [(image_tensor, label_tensor), (image_tensor, label_tensor),...]
            source_batch_img = list(zip(*sources_batch))[0]
            source_batch_class = list(zip(*sources_batch))[1]
            
            # upload to GPU
            source_batch_img = [e.to(device) for e in source_batch_img]
            source_batch_class = [e.to(device) for e in source_batch_class]
            target_batch_img = target_batch_img.to(device)
            
            ## ====================
            # Extract features for source and target
            ## ====================
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                
                logprobs, source_pred, target_pred = mdan(source_batch_img, target_batch_img)
            
                ## ==================== 
                # Compute losses
                ## ==================== 
                # source classification error
                losses = torch.stack([F.nll_loss(logprobs[j], source_batch_class[j]) for j in range(len(sources))])
            
                # domain classification error
                domain_losses = torch.stack([F.nll_loss(source_pred[j], source_labels) +
                                             F.nll_loss(target_pred[j], target_labels) for j in range(len(sources))])
            
                #final_loss = torch.max(losses) + (1e-2) * torch.min(domain_losses)
                final_loss = torch.log(torch.sum(torch.exp(gamma * (losses + mu * domain_losses)))) / gamma
                final_loss.backward()
                optimizer.step()

            running_loss += final_loss.item()
            running_cls_loss += torch.max(losses).item()
            running_domain_loss += torch.max(domain_losses).item()
            iter_cnt+=1
            batch_end_time = time.time() - batch_start_time
            if iter_cnt % train_msg_iter == 0:
                print("Epoch {}/{}, Iteration {}/{}, Training loss: {:.4f}, Time/Batch: {:.4f}".format(
                    epoch,total_epoch,
                    iter_cnt, max_iter,
                    running_loss / (batch_size * 4 * iter_cnt),
                    batch_end_time
                ))
            
            
        print("Epoch {}/{}, Training Loss {:.4f}, Classify Loss: {:.4f}, Domain Loss: {:.4f}".format(
            epoch, total_epoch, 
            running_loss/train_dataset_size,
            running_cls_loss / train_dataset_size,
            running_domain_loss / train_dataset_size
        ))
        train_loss.append(running_loss/train_dataset_size)
        writer.add_scalar('Training_Loss', running_loss/train_dataset_size)
        writer.add_scalar('Classfication_Loss', running_cls_loss/train_dataset_size)
        writer.add_scalar('Domain_Loss', running_domain_loss/train_dataset_size)
        
        ## ========================= 
        # model validation 
        ## =========================
        if epoch % val_model_epoch == 0:
            print("Validating model...")
            validation_loss = 0.0
            validation_corrects = 0
            
            mdan.eval()
            criterion = nn.CrossEntropyLoss()
            
            dataloader = validate_dataloader[target]
            
            # model predict
            for imgs, label in dataloader:
                imgs = imgs.to(device)
                label = label.to(device)
                
                with torch.set_grad_enabled(False):
                    outputs = mdan.inference(imgs)
                    _, preds = torch.max(outputs, 1)
                    loss = F.nll_loss(outputs, label)
                
                validation_loss += loss.item()
                validation_corrects += torch.sum(preds == label.data)
                
            validation_loss = validation_loss / len(validate_datasets[target])
            validation_corrects = validation_corrects.double() / len(validate_datasets[target])
            
            print("Epoch: {}/{}, Validation Loss: {:.4f}, Validation Acc: {:.4f}".format(
                epoch, total_epoch,
                validation_loss, validation_corrects
            ))
            val_acc.append(validation_corrects)
            val_loss.append(validation_loss)
            
            writer.add_scalar('Validation_Loss', validation_loss)
            writer.add_scalar('Validation_Accuracy', validation_corrects)
            
            if validation_corrects >= best_acc:
                print("Update model...")
                best_acc = validation_corrects
                best_model_wts = copy.deepcopy(mdan.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': mdan.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': {'train': train_loss, 'val': val_loss},
                    'acc': {'val': val_acc}
                    }, os.path.join(test_case_place, 'best_model.pt')
                )
