import torch.utils.data as data
import os

from PIL import Image
import constant

class DataGenerator(data.Dataset):
    def __init__(self, dataset_name, train = True, transform=None, class_num = 0):
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
        
        self.classes = [name for name in os.listdir(os.path.join(constant.data_root, dataset_name)) if 'csv' not in name]
        self.classes = sorted(self.classes, key=lambda s: s.lower())

        if class_num > 0:
            pick_class_code = list(range(class_num))
            temp_img_path = []
            temp_img_label = []
            for (img_path, label) in zip(self.img_paths, self.img_labels):
                if label in pick_class_code:
                    temp_img_path.append(img_path)
                    temp_img_label.append(label)
            
            self.img_paths = temp_img_path
            self.img_labels = temp_img_label
            self.classes = self.classes[:class_num]



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


class TestDataGenerator(data.Dataset):
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
        
        data_folder = os.path.join(constant.data_root, dataset_name)
        self.img_paths = [os.path.join(dataset_name, file_name) for file_name in os.listdir(data_folder)]
        

    def __getitem__(self, idx):
        
        img_path = self.img_paths[idx]
        img_path = os.path.join(constant.data_root, img_path)
        
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.img_paths)