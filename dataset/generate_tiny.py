
import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils_new import check, process_dataset, separate_data, split_data, save_file, separate_data_few_shot_iid, separate_data_few_shot_pat_non_iid
from torchvision.datasets import ImageFolder, DatasetFolder

random.seed(1)
np.random.seed(1)

dir_path = "tiny"
if not dir_path.endswith('/'):
    dir_path += '/'
    
num_classes = 200

# http://cs231n.stanford.edu/tiny-imagenet-200.zip
# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


# Allocate data to users
def generate_tiny(dir_path, num_clients, num_classes, niid, balance, partition, alpha, few_shot, n_shot):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition, alpha, few_shot, n_shot):
        return

    transform = transforms.ToTensor()

    trainset = ImageFolder_custom(root=dir_path+'rawdata/tiny-imagenet-200/train/', transform=transform)
    testset = ImageFolder_custom(root=dir_path+'rawdata/tiny-imagenet-200/val/', transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False)

    dataset_image = []
    dataset_label = []

    train_images, train_labels = process_dataset(trainset)
    test_images, test_labels = process_dataset(testset)
    
    dataset_image.extend(train_images)
    dataset_image.extend(test_images)
    dataset_label.extend(train_labels)
    dataset_label.extend(test_labels)
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    
    # print(f'dataset_image: {dataset_image}')
    # print(f'dataset_image length : {len(dataset_image)}')
    # print(f'dataset_label: {dataset_label}')
    # print(f'dataset_label length : {len(dataset_label)}')

    
    if few_shot:  # Add a parameter or a condition to trigger few-shot scenario
        if not niid: # iid
            train_data, test_data, statistic, statistic_test = separate_data_few_shot_iid((dataset_image, dataset_label), 
                                                        num_clients, num_classes, n_shot)
        else: # pat
            train_data, test_data, statistic, statistic_test = separate_data_few_shot_pat_non_iid((dataset_image, dataset_label), 
                                                        num_clients, num_classes, n_shot)
        
    else:

        train_data, test_data, statistic, statistic_test = separate_data((dataset_image, dataset_label), num_clients, num_classes,  
                                    niid, balance, partition, alpha, class_per_client=4)
        # train_data, test_data = split_data(X, y)
        
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, alpha, few_shot, n_shot)


if __name__ == "__main__":
    # Check if the minimum number of arguments is provided
    if len(sys.argv) < 7:
        print("Usage: script.py num_clients niid balance partition alpha few_shot [n_shot]")
        sys.exit(1)

    # Parse arguments
    try:
        num_clients = int(sys.argv[1])
    except ValueError:
        print("Invalid input for num_clients. Please provide an integer value.")
        sys.exit(1)

    niid = sys.argv[2].lower() == "noniid"
    balance = sys.argv[3].lower() == "balance"
    partition = sys.argv[4]
            
    # Alpha is required only for non-IID data with "dir" partition
    alpha = None
    if niid and partition == "dir":
        if len(sys.argv) < 6 or sys.argv[5] == "-":
            print("Alpha parameter is required for non-IID 'dir' partitioned data.")
            sys.exit(1)
        try:
            alpha = float(sys.argv[5])
        except ValueError:
            print("Invalid input for alpha. Please provide a float value.")
            sys.exit(1)
    elif len(sys.argv) >= 6 and sys.argv[5] != "-":
        # Optional alpha for other cases
        try:
            alpha = float(sys.argv[5])
        except ValueError:
            print("Invalid input for alpha. Please provide a float value or '-' for default.")
            sys.exit(1)

    few_shot = sys.argv[6].lower() in ["true", "fs"]

    n_shot = None
    if few_shot:
        if len(sys.argv) < 8:
            print("n_shot parameter is required for few_shot mode.")
            sys.exit(1)
        try:
            n_shot = int(sys.argv[7])
        except ValueError:
            print("Invalid input for n_shot. Please provide an integer value.")
            sys.exit(1)
    
    # Print all parsed arguments
    print(f"Running script with the following parameters:")
    print(f"num_clients: {num_clients}")
    print(f"niid: {niid}")
    print(f"balance: {balance}")
    print(f"partition: {partition}")
    print(f"alpha: {alpha}")
    print(f"few_shot: {few_shot}")
    print(f"n_shot: {n_shot}")

    generate_tiny(dir_path, num_clients, num_classes, niid, balance, partition, alpha, few_shot, n_shot)