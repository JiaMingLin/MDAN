import constant
from datasets import real_loader
from datasets import sketch_loader
from datasets import infograph_loader
from datasets import quickdraw_loader
from datasets import test_loader

import torch
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

LOADER_MAPPING = {
    "rel": real_loader,
    "inf": infograph_loader, 
    "qdr": quickdraw_loader, 
    "skt": sketch_loader,
    "test": test_loader,
}

def load_data(valid_code, is_train = True, resize = None, class_num = 0, trans = True):
    
    loader = LOADER_MAPPING[valid_code]
    dataset = loader.get_dataset(is_train, resize, class_num, trans)
    
    return dataset

def get_sampler(dataset):
    sampling = constant.sampling
    sampling_size = constant.sampling_size
    if sampling == 'random':
        return RandomSampler(dataset, num_samples=sampling_size, replacement=True)

    elif sampling == 'over':
        targets = [int(label) for label in dataset.img_labels]   # in list
        class_count = np.unique(targets, return_counts=True)[1]
        weight = 1. / class_count
        samples_weight = weight[targets]
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, num_samples=sampling_size, replacement = True)

        return sampler


def multiple_data_loader(code_list, batch_size, is_train = True, resize = None, class_num = 0):
    """
    Args:
        Input: 
        Output:
            Sampled batch form multiple dataset
            {
                'skt': batch of (image, label)
                'qdr': batch of (image, label)
                'info': batch of (image, label)
            }
    """
    

    dataloader_mapping = {}

    
    max_data_size = 0
    for code in code_list:
        dataset = load_data(code, is_train, resize, class_num)
        if len(dataset) > max_data_size:
            max_data_size = len(dataset)
        dataloader_mapping[code] = dataset

    # number of batches
    #batch_num = int(max_data_size/batch_size) + 1
    batch_num = int(constant.sampling_size / batch_size) + 1
    for code in code_list:
        dataset = dataloader_mapping[code]
        
        # random sampler, replacement = True
        #rand_sampler = RandomSampler(dataset, num_samples=batch_size*batch_num, replacement=True)
        rand_sampler = get_sampler(dataset)
        data_sampler = DataLoader(dataset, batch_size=batch_size, sampler=rand_sampler, num_workers = constant.num_workers)
        dataloader_mapping[code] = iter(data_sampler)
        
    for batch_idx in range(batch_num):
        batch = {}
        for code in code_list:
            loader = dataloader_mapping[code]
            imgs, labels = next(loader)
            batch[code] = (imgs, labels)
        yield batch