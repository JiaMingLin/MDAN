import os
base_path = os.getcwd()
data_root = os.path.join(base_path, 'data/')
logs_root = os.path.join(base_path, 'logs/')

dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)

num_workers = 8