#!/usr/bin/env python
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

"""
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="amazon")
parser.add_argument("-f", "--frac", help="Fraction of the supervised training data to be used.",
                    type=float, default=1.0)
parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
parser.add_argument("-v", "--verbose", help="Verbose mode: True -- show training progress. False -- "
                                            "not show training progress.", type=bool, default=True)
parser.add_argument("-m", "--model", help="Choose a model to train: [mdan]",
                    type=str, default="mdan")
# The experimental setting of using 5000 dimensions of features is according to the papers in the literature.
parser.add_argument("-d", "--dimension", help="Number of features to be used in the experiment",
                    type=int, default=5000)
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                    type=float, default=1e-2)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=15)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=20)
parser.add_argument("-o", "--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str, default="dynamic")
# Compile and configure all the model parameters.
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = get_logger(args.name)  # amazon

# Set random number seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)

## ==============================
# Preparing Dataset
## ==============================
image_size = (224,224)

logger.info("Data sets: {}".format(data_name))
logger.info("Number of total instances in the data sets: {}".format(num_insts))


logger.info("-" * 100)
"""
num_data_sets = 4
for i in range(num_data_sets):
    ## ========================= 
    # Build source instances.
    ## ========================= 
    
    ## ========================= 
    # Build target instances.
    ## ========================= 

    ## ========================= 
    # Train MDAN.
    ## ========================= 
    mdan = MDANet(configs).to(device)
    optimizer = optim.Adadelta(mdan.parameters(), lr=lr)
    mdan.train()
    # Training phase.
    time_start = time.time()
    for t in range(num_epochs):
        running_loss = 0.0

        # one batch, xs for all source domains, ys is the label
        train_loader = dl.multiple_data_loader(['rel', 'sketch', 'inf'],32, resize = (224,224))

"""
            for xs, ys in train_loader:
                slabels = torch.ones(batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                tlabels = torch.zeros(batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                for j in range(num_domains):
                    xs[j] = torch.tensor(xs[j], requires_grad=False).to(device)
                    ys[j] = torch.tensor(ys[j], requires_grad=False).to(device)
                ridx = np.random.choice(target_insts.shape[0], batch_size)
                tinputs = target_insts[ridx, :]
                tinputs = torch.tensor(tinputs, requires_grad=False).to(device)
                optimizer.zero_grad()
                logprobs, sdomains, tdomains = mdan(xs, tinputs)
                # Compute prediction accuracy on multiple training sources.
                losses = torch.stack([F.nll_loss(logprobs[j], ys[j]) for j in range(num_domains)])
                domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) +
                                           F.nll_loss(tdomains[j], tlabels) for j in range(num_domains)])
                # Different final loss function depending on different training modes.
                if mode == "maxmin":
                    loss = torch.max(losses) + mu * torch.min(domain_losses)
                elif mode == "dynamic":
                    loss = torch.log(torch.sum(torch.exp(gamma * (losses + mu * domain_losses)))) / gamma
                else:
                    raise ValueError("No support for the training mode on madnNet: {}.".format(mode))
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            logger.info("Iteration {}, loss = {}".format(t, running_loss))
        time_end = time.time()
        # Test on other domains.
        mdan.eval()
        target_insts = torch.tensor(target_insts, requires_grad=False).to(device)
        target_labels = torch.tensor(target_labels)
        #preds_labels = torch.max(mdan.inference(target_insts), 1)[1].cpu().data.squeeze_()
        preds_labels = torch.max(mdan.inference(target_insts), 1)[1].cpu().data
        pred_acc = torch.sum(preds_labels == target_labels).item() / float(target_insts.size(0))
        error_dicts[data_name[i]] = preds_labels.numpy() != target_labels.numpy()
        logger.info("Prediction accuracy on {} = {}, time used = {} seconds.".
                    format(data_name[i], pred_acc, time_end - time_start))
        results[data_name[i]] = pred_acc
    logger.info("Prediction accuracy with multiple source domain adaptation using madnNet: ")
    logger.info(results)
    pickle.dump(error_dicts, open("{}-{}-{}-{}.pkl".format(args.name, args.frac, args.model, args.mode), "wb"))
    logger.info("*" * 100)
else:
    raise ValueError("No support for the following model: {}.".format(args.model))
"""
