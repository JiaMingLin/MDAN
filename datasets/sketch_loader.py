import torch
from torchvision import datasets, transforms
from .datagen import DataGenerator
import constant

def get_dataset(train, resize = None, class_num = 0):
    """Get real dataset loader."""
    
    name = 'sketch'
    # image pre-processing
    transform_pipline = []
    if resize is not None:
        transform_pipline.append(transforms.Resize(resize))
    """
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
    """
    transform_pipline.append(transforms.ToTensor())
    transform_pipline.append(transforms.Normalize(mean= constant.dataset_mean, std = constant.dataset_std))
    transform = transforms.Compose(transform_pipline)
    
    # dataset and data loader
    dataset = DataGenerator(
        dataset_name = name,
        train = train,
        transform = transform,
        class_num = class_num
    )

    return dataset

if __name__ == "__main__":
    get_dataset(True)