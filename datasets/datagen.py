import torch.utils.data as data
import os

from PIL import Image
import constant

class DataGenerator(data.Dataset):
    def __init__(self, dataset_name, train = True, transform=None):
        """
        Args:
            1. image folder
            2. data name, label list
            3. if train, loading data from train folder, or test folder
            4. 
        """
        self.dataset_name = dataset_name
        self.transform = transform
        self.train = train
        
        
        if train:
            data_list = os.path.join(constant.data_root, dataset_name, '{}_train.csv'.format(dataset_name))
        else:
            data_list = os.path.join(constant.data_root, dataset_name, '{}_test.csv'.format(dataset_name))

        with open(data_list) as fin:
            data_list = fin.readlines()

        self.img_paths = []
        self.img_labels = []

        self.n_data = 0
        for data in data_list[1:]:
            data = data.strip('\n').split(',')
            self.img_paths.append(data[0])
            self.img_labels.append(data[1])
            self.n_data += 1

    def __getitem__(self, idx):
        
        img_path, label = self.img_paths[idx], self.img_labels[idx]
        if self.train is True:
            img_path = os.path.join(constant.data_root, img_path)
        else:
            img_path = os.path.join(constant.data_root, img_path)
        
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            label = int(label)

        return img, label

    def __len__(self):
        return len(self.img_paths)