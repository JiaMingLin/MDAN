import constant
from datasets import real_loader
from datasets import sketch_loader
from datasets import infograph_loader
from datasets import quickdraw_loader

import torch
import torch.utils.data as data
import os
from PIL import Image

LOADER_MAPPING = {
    "rel": real_loader,
    "inf": infograph_loader, 
    "qdr": quickdraw_loader, 
    "skt": sketch_loader
}

def load_data(valid_code, is_train = True, resize = None):
    
    loader = LOADER_MAPPING[valid_code]

    dataset = loader.get_dataset(is_train, resize)
    
    return dataset