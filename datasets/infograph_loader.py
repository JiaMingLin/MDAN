import torch
from torchvision import datasets, transforms
from .datagen import DataGenerator
import constant

def get_dataset(train):
    """Get real dataset loader."""
    
    name = 'infograph'
    # image pre-processing
    
    if train:
        transform = transforms.Compose([
            #transforms.RandomCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=constant.dataset_mean, std=constant.dataset_std)
        ])
    else:
        transform = transforms.Compose([
            #transforms.CenterCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= constant.dataset_mean, std = constant.dataset_std)
        ])
        
    # dataset and data loader
    dataset = DataGenerator(
        dataset_name = name,
        train = train,
        transform = transform
    )

    return dataset

if __name__ == "__main__":
    get_dataset(True)