import constant
from datasets import real_loader
from datasets import sketch_loader
from datasets import infograph_loader
from datasets import quickdraw_loader
from datasets import test_loader

import torch
import torch.utils.data as data
import os
from PIL import Image
from torch.utils.data import DataLoader

LOADER_MAPPING = {
    "rel": real_loader,
    "inf": infograph_loader, 
    "qdr": quickdraw_loader, 
    "skt": sketch_loader,
    "test": test_loader,
}

def load_data(valid_code, is_train = True, resize = None, class_num = 0):
    
    loader = LOADER_MAPPING[valid_code]
    dataset = loader.get_dataset(is_train, resize, class_num)
    
    return dataset

def multiple_data_loader(code_list, batch_size, is_train = True, resize = None, class_num = 0):
    # dataset
    max_data_size = 0

    dataloader_mapping = {}

    for code in code_list:
        dataset = load_data(code, is_train, resize, class_num)
        if len(dataset) > max_data_size:
            max_data_size = len(dataset)
        # loader
        loader = DataLoader(dataset, batch_size=batch_size,shuffle=is_train, num_workers=constant.num_workers)
        dataloader_mapping[code] = iter(loader)

    # number of batches
    batch_num = int(max_data_size/batch_size) + 1
    
    for    batch_idx in range(batch_num):
        multi_batch = {}
        for code in code_list:
            loader = dataloader_mapping[code]
            imgs, labels = next(loader)
            multi_batch[code] = (imgs, labels)

        yield multi_batch
