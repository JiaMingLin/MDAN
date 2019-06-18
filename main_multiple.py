#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import constant
from utils import get_logger, get_lr, save_model, resume_checkpoint
from datasets import data_loader as dl
from models.model_fectory import MDANet, load_model
from models.warmup_scheduler import GradualWarmupScheduler

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import shutil
import read_config

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
parser.add_argument('--config_file_path', type=str, default='conf_files/config')

## ===============================
# Compile and configure all the model parameters.
## ===============================
args = parser.parse_args()
config = read_config.read(args.config_file_path)
print(config)
name = config['name']
total_epoch = config['total_epoch']
batch_size = config['batch_size']
class_number = config['class_number']
learning_rate = config['learning_rate']
extractor = config['extractor']
image_size = config['image_size']
resume_train = bool(config['resume'])
constant.number_worker = config['number_workers']
constant.sampling = config['sampling']
constant.sampling_size = config['sampling_size']
seed = 42
gamma = 10
mu = 1e-2
val_model_epoch = 3
train_msg_iter = 100
resize = (image_size,image_size)

## ===============================
# Saved result place
## ===============================

test_case_place = os.path.join(constant.logs_root, name)

# for new training, create new save place
# if resume, keep previous save place
if resume_train is False:
    if os.path.isdir(test_case_place):
        shutil.rmtree(test_case_place)
    os.makedirs(test_case_place)

logger = get_logger(os.path.join(test_case_place, 'messages'))
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
    'rel': DataLoader(validate_datasets['rel'], batch_size=batch_size,shuffle=False, num_workers=constant.number_worker),
    'skt': DataLoader(validate_datasets['skt'], batch_size=batch_size,shuffle=False, num_workers=constant.number_worker),
    'qdr': DataLoader(validate_datasets['qdr'], batch_size=batch_size,shuffle=False, num_workers=constant.number_worker),
    'inf': DataLoader(validate_datasets['inf'], batch_size=batch_size,shuffle=False, num_workers=constant.number_worker)
}

datasets = ['rel', 'skt', 'qdr', 'inf']
#for target in datasets:
for target in ['rel']:
    sources = list(filter(lambda e: e != target, datasets))
    logger.info("Selected sources: {}".format(str(sources)))
    logger.info("Selected target: {}".format(target))
    logger.info("="*100)
    ## ==========================
    # Initialize MDAN model
    ## ==========================
    mdan = load_model('mdan', class_number, len(sources), extractor).to(device)
    optimizer = optim.Adadelta(mdan.parameters(), lr=learning_rate)
    # Decay LR by a factor of 0.1 every 7 epochs
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    resume_epoch = 0
    if resume_train is True:
        resume_epoch, model_state_dict, optimizer_state_dict = resume_checkpoint(test_case_place, file_name = 'best_model.pt')
        mdan.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        mdan.eval()
        logger.info("Retain training from epoch {}".format(resume_epoch))
    else:
        mdan.train()

    scheduler_plateau = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=17, after_scheduler=scheduler_plateau)
    
    ## ==========================
    #  MDAN Training
    ## ==========================
    max_iter = int(constant.sampling_size/batch_size) + 1
    best_acc = 0
    train_loss = []
    val_acc = []
    val_loss = []
    logger.info("Total Epochs: {}, Total iteration: {}/epoch".format(total_epoch, max_iter))
    logger.info("Starting training...")
    
    for epoch in range(resume_epoch, total_epoch):
        
        running_loss = 0.0
        running_cls_loss = 0.0
        running_domain_loss = 0.0
        ## ==========================
        # Sub-Sampling data with replacement = True
        ## ==========================
        multi_data_loader = dl.multiple_data_loader(
            datasets, batch_size, is_train = True, resize = resize, class_num = class_number)
        
        iter_cnt = 0
        epoch_start_time = time.time()
        for batch in multi_data_loader:
            
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
            
            """
            if iter_cnt % train_msg_iter == 0:
                print("Epoch {}/{}, Iteration {}/{}, Training loss: {:.4f}, Time/Batch: {:.4f}".format(
                    epoch,total_epoch,
                    iter_cnt, max_iter,
                    running_loss / (batch_size * 4 * iter_cnt),
                    batch_end_time
                ))
            """
            
        epoch_end_time = time.time() - epoch_start_time    
        logger.info("Epoch {}/{}, Training Loss {:.4f}, Classify Loss: {:.4f}, Domain Loss: {:.4f}, Time/Epoch: {:.4f}".format(
            epoch, total_epoch, 
            running_loss/train_dataset_size,
            running_cls_loss / train_dataset_size,
            running_domain_loss / train_dataset_size,
            epoch_end_time
        ))
        train_loss.append(running_loss/train_dataset_size)
        writer.add_scalar('Training_Loss', running_loss/train_dataset_size)
        writer.add_scalar('Classfication_Loss', running_cls_loss/train_dataset_size)
        writer.add_scalar('Domain_Loss', running_domain_loss/train_dataset_size)
        writer.add_scalar('Learning_Rate', get_lr(optimizer))

        # 
        scheduler_warmup.step(epoch, metrics=(running_loss/train_dataset_size))
        
        ## ========================= 
        # model validation 
        ## =========================
        if epoch % val_model_epoch == 0:
            logger.info("Validating model...")
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
            
            logger.info("Epoch: {}/{}, Validation Loss: {:.4f}, Validation Acc: {:.4f}".format(
                epoch, total_epoch,
                validation_loss, validation_corrects
            ))
            val_acc.append(validation_corrects)
            val_loss.append(validation_loss)
            
            writer.add_scalar('Validation_Loss', validation_loss)
            writer.add_scalar('Validation_Accuracy', validation_corrects)
            save_model(
                model_name = 'checkpoint_epoch_{}.pt'.format(epoch), 
                epoch = epoch, 
                model = mdan, 
                optimizer = optimizer,
                loss = {'train': train_loss, 'val': val_loss}, 
                acc = {'val': val_acc}, 
                save_dir = test_case_place
            )

            
            if validation_corrects >= best_acc:
                logger.info("Update model to path {}...".format(os.path.join(test_case_place, 'best_model.pt')))
                best_acc = validation_corrects
                save_model(
                    model_name = 'best_model.pt', 
                    epoch = epoch, 
                    model = mdan, 
                    optimizer = optimizer,
                    loss = {'train': train_loss, 'val': val_loss}, 
                    acc = {'val': val_acc}, 
                    save_dir = test_case_place
                )
                
