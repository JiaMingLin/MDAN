import torch
from torchvision import datasets, transforms
from .datagen import DataGenerator
from .cust_transforms import Lighting, __imagenet_pca
import constant

def get_dataset(train, resize = None, class_num = 0):
    """Get real dataset loader."""
    
    name = 'infograph'
    # image pre-processing
    transform_pipline = []
    """
    if resize is not None:
        transform_pipline.append(transforms.Resize(resize))
    """
    if train:
        transform_pipline = transform_pipline + [
            transforms.RandomResizedCrop(size = resize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=[0.6,1.4], contrast=[0.6,1.4], saturation=[0.6,1.4], hue=[-.5, .5]),
            transforms.ToTensor(),
            transforms.Normalize(mean=constant.dataset_mean, std=constant.dataset_std),
            Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec'])
        ]
    else:
        transform_pipline = transform_pipline + [
            transforms.Resize(size = resize[0]+32),#transforms.CenterCrop((28)),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean= constant.dataset_mean, std = constant.dataset_std)
        ]

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