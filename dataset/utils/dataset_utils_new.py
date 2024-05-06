# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split

batch_size = 128
train_size = 0.75 # merge original training set and test set, then split it manually. 
least_samples = 1 # guarantee that each client must have at least one samples for testing.

def check(config_path, train_path, test_path, num_clients, num_classes, niid=False, 
        balance=False, partition=None, alpha=None, few_shot=False, n_shot=None, pfl=False):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['num_classes'] == num_classes and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size and \
            config['few_shot'] == few_shot and \
            config['n_shot'] == n_shot and \
            config['pfl'] == pfl:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def process_dataset(data):
    images = []
    labels = []

    for image, label in data:
        np_image = np.array(image)
        # Check if the image has a single channel (1, 224, 224)
        if np_image.shape[0] == 1:
            # Convert it to a three-channel image by repeating the single channel 3 times
            np_image = np.repeat(np_image, 3, axis=0)
        # Append the image and label to their respective lists
        images.append(np_image)
        labels.append(label)

    # Stack the images along a new axis
    images_array = np.stack(images)
    labels_array = np.array(labels)

    return images_array, labels_array


def separate_data_few_shot_iid(data, num_clients, num_classes, n_shot):
    train_data = []
    test_data = []
    statistic_train = [[] for _ in range(num_clients)]
    statistic_test = []

    dataset_content, dataset_label = data

    # Split the dataset into common testing set and training pool
    idxs = np.arange(len(dataset_label))
    test_size = int(len(dataset_content) * 0.20)
    test_idxs = np.random.choice(idxs, size=test_size, replace=False)
    train_idxs = np.setdiff1d(idxs, test_idxs)

    X_test = dataset_content[test_idxs]
    y_test = dataset_label[test_idxs]
    
    # Compute statistics for the testing data
    for i in np.unique(y_test):
        statistic_test.append((int(i), int(np.sum(y_test == i))))

    # Allocate n-shot samples per class to each client from the training pool
    for client in range(num_clients):
        X_train_client = []
        y_train_client = []

        for i in range(num_classes):
            class_idxs = np.where(dataset_label[train_idxs] == i)[0]
            np.random.shuffle(class_idxs)
            if len(class_idxs) < n_shot:
                raise ValueError(f"Not enough samples to allocate {n_shot} samples per class to each client.")
            
            selected_idxs = class_idxs[:n_shot]
            X_train_client.extend(dataset_content[train_idxs][selected_idxs])
            y_train_client.extend(dataset_label[train_idxs][selected_idxs])

        # Convert training data to numpy arrays
        X_train_client = np.array(X_train_client)
        y_train_client = np.array(y_train_client)

        # Update training statistics for the client after all classes are processed
        for label in np.unique(y_train_client):
            statistic_train[client].append((int(label), int(np.sum(y_train_client == label))))

        train_data.append({'x': X_train_client, 'y': y_train_client})

    # Package the test data similarly for all clients
    for _ in range(num_clients):
        test_data.append({'x': X_test, 'y': y_test})

    # Print dataset details for each client and the common testing set
    for client in range(num_clients):
        print(f"Client {client}\t Size of training data: {len(train_data[client]['x'])}\t Training labels: ", np.unique(train_data[client]['y']))
        print(f"\t\t Training samples of labels: ", [i for i in statistic_train[client]])
        print("-" * 50)
    print(f"Size of common testing data: {len(X_test)}\t Testing labels: ", np.unique(y_test))
    print(f"\t Testing samples of labels: ", [i for i in statistic_test])

    return train_data, test_data, statistic_train, statistic_test




def separate_data_few_shot_pat_non_iid(data, num_clients, num_classes, n_shot):
    k = num_classes // num_clients  # Base number of classes per client
    remainder = num_classes % num_clients  # Remainder classes to be added to the last client

    train_data = []
    test_data = []
    statistic_train = [[] for _ in range(num_clients)]
    statistic_test = []

    dataset_content, dataset_label = data

    # Split the dataset into common testing set and training pool
    idxs = np.arange(len(dataset_label))
    test_size = int(len(dataset_content) * 0.20)
    test_idxs = np.random.choice(idxs, size=test_size, replace=False)
    train_idxs = np.setdiff1d(idxs, test_idxs)

    X_test = dataset_content[test_idxs]
    y_test = dataset_label[test_idxs]

    # Compute statistics for the testing data
    for i in np.unique(y_test):
        statistic_test.append((int(i), int(np.sum(y_test == i))))

    # Shuffle class indices to distribute them randomly among clients
    class_indices = np.arange(num_classes)
    np.random.shuffle(class_indices)

    start_idx = 0
    # Allocate n-shot training samples for k classes to each client
    for client in range(num_clients):
        X_train_client = []
        y_train_client = []

        # Determine the number of classes this client will have
        num_classes_client = k + (remainder if client == num_clients - 1 and remainder > 0 else 0)
        client_classes = class_indices[start_idx:start_idx + num_classes_client]
        start_idx += num_classes_client

        for class_id in client_classes:
            class_idxs = np.where(dataset_label[train_idxs] == class_id)[0]
            np.random.shuffle(class_idxs)

            if len(class_idxs) < n_shot:
                raise ValueError(f"Not enough samples to allocate {n_shot} samples per class to client {client}.")

            selected_idxs = class_idxs[:n_shot]
            X_train_client.extend(dataset_content[train_idxs][selected_idxs])
            y_train_client.extend(dataset_label[train_idxs][selected_idxs])

        # Update statistics after collecting all class data for the client
        y_train_client_array = np.array(y_train_client)
        for label in np.unique(y_train_client_array):
            statistic_train[client].append((int(label), int(np.sum(y_train_client_array == label))))

        train_data.append({'x': np.array(X_train_client), 'y': y_train_client_array})


    # Package the test data similarly for all clients
    for _ in range(num_clients):
        test_data.append({'x': X_test, 'y': y_test})

    # Print dataset details for each client and the common testing set
    for client in range(num_clients):
        print(f"Client {client}\t Size of training data: {len(train_data[client]['x'])}\t Training labels: ", np.unique(train_data[client]['y']))
        print(f"\t\t Training samples of labels: ", [i for i in statistic_train[client]])
        print("-" * 50)
    print(f"Size of common testing data: {len(X_test)}\t Testing labels: ", np.unique(y_test))
    print(f"\t Testing samples of labels: ", [i for i in statistic_test])

    return train_data, test_data, statistic_train, statistic_test



def separate_data_pfl(data, num_clients, num_classes, niid=False, balance=False, partition=None, alpha=None, class_per_client=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)*class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1


    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
            

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, alpha=None, class_per_client=None):
    X_train = [[] for _ in range(num_clients)]
    y_train = [[] for _ in range(num_clients)]
    train_data = []
    test_data = []
    statistic_train = [[] for _ in range(num_clients)]
    statistic_test = []

    dataset_content, dataset_label = data
    
    print(f'dataset_label: {len(dataset_label)}')

    # Split the dataset into common testing set and training pool
    idxs_total = np.arange(len(dataset_label))
    test_size = int(len(dataset_content) * 0.20)
    test_idxs = np.random.choice(idxs_total, size=test_size, replace=False)
    train_idxs = np.setdiff1d(idxs_total, test_idxs)
    
    
    train_size = len(dataset_content) - test_size
    assert len(train_idxs) == train_size
    assert len(test_idxs) == test_size
    

    X_test = dataset_content[test_idxs]
    y_test = dataset_label[test_idxs]
    
    
    y_train_full = dataset_label[train_idxs]
    
    print(f'y_train_full: {len(y_train_full)}')

    
    # Compute statistics for the testing data
    for i in np.unique(y_test):
        statistic_test.append((int(i), int(np.sum(y_test == i))))

    # Prepare test data for all clients
    common_test_data = {'x': X_test, 'y': y_test}
    test_data.append({'x': X_test, 'y': y_test})

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(y_train_full)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[y_train_full == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)*class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1



    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(y_train_full)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(y_train_full == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
        
    else:
        raise NotImplementedError
    
    
    # Convert training data to numpy arrays and package it
    for client in range(num_clients):
        client_indices = dataidx_map[client]
        X_train[client] = dataset_content[train_idxs][client_indices]
        y_train[client] = y_train_full[client_indices]
        
        # Convert to numpy arrays and add to the train_data list
        X_train[client] = np.array(X_train[client])
        y_train[client] = np.array(y_train[client])
        train_data.append({'x': X_train[client], 'y': y_train[client]})

        # Update statistics for training data
        for label in np.unique(y_train[client]):
            statistic_train[client].append((int(label), int(np.sum(y_train[client] == label))))

        # Print dataset details for each client
        print(f"Client {client}\t Size of training data: {len(X_train[client])}\t Labels: ", np.unique(y_train[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic_train[client]])
        print("-" * 50)
        
    # Print testing data statistics
    print(f"Common testing data size: {len(X_test)}")
    print(f"Testing labels: ", np.unique(y_test))
    print(f"Testing samples of labels: ", [i for i in statistic_test])

    return train_data, test_data, statistic_train, statistic_test



def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
                num_classes, statistic, niid=False, balance=False, partition=None, alpha=None, few_shot=False, n_shot=None, pfl=False):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'alpha': alpha, 
        'batch_size': batch_size, 
        'few_shot': few_shot,
        'n_shot': n_shot,
        'pfl': pfl,
        'Size of samples for labels in clients': statistic, 
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")