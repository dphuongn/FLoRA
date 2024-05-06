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

import time
import copy
import random
from flcore.clients.clientdylora import clientDYLORA
from flcore.servers.serverbase import Server
from threading import Thread
import statistics
import torch
from utils.data_utils import read_client_data_clip
import numpy as np
import os
import json

from flcore.trainmodel.clip_model import *


class FedDyLora(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        
        self.normalized_rank = args.normalized_rank
        self.normalized_alpha = args.normalized_alpha
        
        self.random_rank = args.random_rank
        self.random_alpha = args.random_alpha
        
        self.fixed_rank = args.fixed_rank
        self.fixed_alpha = args.fixed_alpha
        
        
        self.dataset = args.dataset
        self.class_names = args.class_names
        
        self.pwd = os.getcwd()
        
        self.config_path = f'{self.pwd}/../dataset/{self.dataset}/config.json'
        
        with open(self.config_path) as file:
            config = json.load(file)

        # Extract data
        self.num_classes = args.num_classes
        self.sample_data = config['Size of samples for labels in clients']
        
        self.lora_params = args.lora_params
        
        # Temporary CLIP model for initial processor setup
        temp_clip_model = CLIPModelFFT(model_id=args.model_id, home_dir=args.home_dir).to(args.device)
        self.processor = temp_clip_model.processor
        
        # Personalized FL setup
        self.pfl = args.personalized_fl
        self.uniform_weight = args.uniform_weight
        
        # Configure the global model with the maximum rank determined
        global_lora_params = copy.deepcopy(self.lora_params)
        
        
        self.random_ranks = [] # This will store the random ranks
        self.random_alphas = [] # This will store the random alphas
        
        self.train_samples = []
        self.total_train_samples = 0
        self.calculate_total_train_samples()
        
        if self.random_rank:
            for i in range(self.num_clients):
                self.random_rank = random.randint(self.lora_params['rank_min']+1, self.lora_params['rank_max'])
                self.random_ranks.append(self.random_rank)
            
            global_lora_params['rank'] = max(self.random_ranks)
            global_lora_params['alpha'] = self.lora_params['alpha']
        
        elif self.fixed_rank:
            global_lora_params['rank'] = self.lora_params['rank']
                
        else:
            self.adjusted_ranks = []  # This will store the adjusted ranks
            self.normalized_adjusted_ranks = []  # This will store the normalized adjusted ranks
            self.calculate_kl_divergences()  # Calculate KL divergence scores
            
            # Initialize the global CLIP model with the maximum rank determined
            if self.normalized_rank:
                global_lora_params['rank'] = max(self.normalized_adjusted_ranks)
            else:
                global_lora_params['rank'] = max(self.adjusted_ranks)
                 
            
        if self.random_alpha:
            for i in range(self.num_clients):
                self.random_alpha = random.randint(self.lora_params['alpha_min']+1, self.lora_params['alpha_max'])
                self.random_alphas.append(self.random_alpha)
                
            global_lora_params['alpha'] = max(self.random_alphas)
            
        elif self.fixed_alpha:
            global_lora_params['alpha'] = self.lora_params['alpha']
                
        else:
            self.adjusted_alphas = []  # This will store the adjusted alphas
            self.normalized_adjusted_alphas = []  # This will store the normalized adjusted alphas
            self.calculate_data_proportion() # Calculate data proportions
            
            # Initialize the global CLIP model with the maximum alpha determined
            if self.normalized_alpha:
                global_lora_params['alpha'] = max(self.normalized_adjusted_alphas)
            else:
                global_lora_params['alpha'] = max(self.adjusted_alphas)
        
        

        
        # Then initialize clients with the adjusted ranks
        self.initialize_clients(clientDYNAMICLORA)

        # self.set_clients(clientDYNAMICLORA)  # Now set clients with adjusted ranks
            
        print(f'global_lora_params: {global_lora_params}')
            
        self.clip_model_object = CLIPModelWithLoRA(model_id=args.model_id, home_dir=args.home_dir, lora_params=global_lora_params).to(args.device)
            
        self.global_model = copy.deepcopy(self.clip_model_object.get_lora_state_dict())

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        
    def initialize_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            if self.pfl:
                test_data = read_client_data_clip(self.dataset, i, self.processor, self.class_names, self.device, is_train=False)
            else:
                test_data = read_client_data_clip(self.dataset, 0, self.processor, self.class_names, self.device, is_train=False)
            
            print(f'train_samples: {self.train_samples[i]}')
            
            # Use the adjusted LoRA rank from previous calculations
            dynamic_lora_params = copy.deepcopy(self.lora_params)
            
            if self.random_rank:
                dynamic_lora_params['rank'] = self.random_ranks[i]
            elif self.fixed_rank:
                dynamic_lora_params['rank'] = self.lora_params['rank']
            else:
                if self.normalized_rank:
                    dynamic_lora_params['rank'] = self.normalized_adjusted_ranks[i]
                else:
                    dynamic_lora_params['rank'] = self.adjusted_ranks[i]
                    
            if self.random_alpha:
                dynamic_lora_params['alpha'] = self.random_alphas[i]
            elif self.fixed_alpha:
                dynamic_lora_params['alpha'] = self.lora_params['alpha']
            else:
                if self.normalized_alpha:
                    dynamic_lora_params['alpha'] = self.normalized_adjusted_alphas[i]
                else:
                    dynamic_lora_params['alpha'] = self.adjusted_alphas[i]
            
            print(f'dynamic_lora_params: {dynamic_lora_params}')
            
            client = clientObj(self.args,
                            id=i, 
                            train_samples=self.train_samples[i], 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow,
                            lora_params=dynamic_lora_params)
            
            self.clients.append(client)
            
    
    def calculate_total_train_samples(self):
        for i in range(self.num_clients):
            train_data = read_client_data_clip(self.dataset, i, self.processor, self.class_names, self.device, is_train=True)
            self.train_samples.append(len(train_data))
            self.total_train_samples += len(train_data)
    
    def calculate_data_proportion(self):
        
        print(f'total_train_samples: {self.total_train_samples}')
        
        data_proportions = [train_samples / self.total_train_samples for train_samples in self.train_samples]
        
        print(f'data_proportions: {data_proportions}')
        
        max_dp = max(data_proportions)
        min_dp = min(data_proportions)
        normalized_data_proportions = [(dp - min_dp) / (max_dp - min_dp) if max_dp != min_dp else 0.5 for dp in data_proportions]
        
        print(f'normalized_data_proportions: {normalized_data_proportions}')
        
        self.adjusted_alphas = [
            max(1, round(self.lora_params['alpha_min'] + (self.lora_params['alpha_max'] - self.lora_params['alpha_min']) * data_proportion))
            for data_proportion in data_proportions
        ]
        
        self.normalized_adjusted_alphas = [
            max(1, round(self.lora_params['alpha_min'] + (self.lora_params['alpha_max'] - self.lora_params['alpha_min']) * normalized_data_proportion))
            for normalized_data_proportion in normalized_data_proportions
        ]
    
    def calculate_kl_divergences(self):
        total_prob, client_probs = self.calculate_client_distributions()
        
        kl_divergences = [self.kl_divergence(client_prob, total_prob) for client_prob in client_probs]
        
        max_kl = max(kl_divergences)
        min_kl = min(kl_divergences)
        normalized_kl_divergences = [(kl - min_kl) / (max_kl - min_kl) if max_kl != min_kl else 0.5 for kl in kl_divergences]
        
        print(f'kl_divergences: {kl_divergences}')
        print(f'normalized_kl_divergences: {normalized_kl_divergences}')
        
        total_variance = sum(kl_divergences)
        
        normalized_total_variance = sum(normalized_kl_divergences)
        
        self.adjusted_ranks = [
            max(1, round(self.lora_params['rank_min'] + (self.lora_params['rank_max'] - self.lora_params['rank_min']) * (kl_div / total_variance)))
            for kl_div in kl_divergences
        ]
        
        self.normalized_adjusted_ranks = [
            max(1, round(self.lora_params['rank_min'] + (self.lora_params['rank_max'] - self.lora_params['rank_min']) * (normalized_kl_div / normalized_total_variance)))
            for normalized_kl_div in normalized_kl_divergences
        ]
                
    def kl_divergence(self, p, q):
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        return np.sum(p * np.log(p / q))
    
    # Function to calculate distributions
    def calculate_client_distributions(self):
        total_counts = np.zeros(self.num_classes)
        client_distributions = []

        for client_data in self.sample_data:
            client_counts = np.zeros(self.num_classes)
            for class_id, count in client_data:
                client_counts[class_id] += count
                total_counts[class_id] += count
            client_distributions.append(client_counts)

        total_prob = total_counts / np.sum(total_counts)
        client_probs = [client_count / np.sum(client_count) for client_count in client_distributions]

        return total_prob, client_probs
    
    # def set_clients(self, clientObj):
        
        total_samples = self.calculate_total_train_samples()
        
        print(f'total_samples: {total_samples}')
        
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data_clip(self.dataset, i, self.processor, self.class_names, self.device, is_train=True)
            
            if self.pfl:
                test_data = read_client_data_clip(self.dataset, i, self.processor, self.class_names, self.device, is_train=False)
            else:
                test_data = read_client_data_clip(self.dataset, 0, self.processor, self.class_names, self.device, is_train=False)
                
            train_samples = len(train_data)
            
            print(f'train_samples: {train_samples}')
            
            # Calculate the proportion of data and determine dynamic rank
            data_proportion = train_samples / total_samples
            dynamic_rank_round = round(self.lora_params['rank_min'] + (self.lora_params['rank_max'] - self.lora_params['rank_min']) * data_proportion)
            
            dynamic_rank = max(1, dynamic_rank_round)
            
            # Keep track of the maximum rank used
            self.global_max_rank = max(self.global_max_rank, dynamic_rank)

            dynamic_lora_params = copy.deepcopy(self.lora_params)
            dynamic_lora_params['rank'] = dynamic_rank
            
            print(f'dynamic_lora_params: {dynamic_lora_params}')
            
            client = clientObj(self.args,
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow,
                            lora_params=dynamic_lora_params)
            self.clients.append(client)


    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                # client.
                client.train()
                
                

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
            
        
    # added ------------------------------------------------------
    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        
        # Find the index of the model with the largest rank
        max_rank_index = 0
        max_rank_size = 0
        for i, model in enumerate(self.uploaded_models):
            # Check both potential rank dimensions due to varying shapes
            for key, value in model.items():
                current_rank_size = value.size(1) if int(key.split('_')[-1]) % 2 == 0 else value.size(0)
                if current_rank_size > max_rank_size:
                    max_rank_size = current_rank_size
                    max_rank_index = i
        
        
        # Initialize global LoRA parameters using the model with the biggest rank
        self.global_model = {k: torch.zeros_like(v) for k, v in self.uploaded_models[max_rank_index].items()}

        # Aggregate LoRA parameters from each client model
        for weight, client_model in zip(self.uploaded_weights, self.uploaded_models):
            for param_key in self.global_model.keys():
                client_param = client_model[param_key]
                param_index = int(param_key.split('_')[-1])  # Determine even or odd index
                
                # test for no weight
                if self.uniform_weight is True:
                    weight = 1.0

                if param_index % 2 == 0:
                    # Even index, adjust the second dimension
                    if client_param.shape[1] < self.global_model[param_key].shape[1]:
                        padded_param = torch.zeros_like(self.global_model[param_key])
                        padded_param[:, :client_param.shape[1]] = client_param
                        self.global_model[param_key] += padded_param * weight
                    else:
                        self.global_model[param_key] += client_param * weight
                else:
                    # Odd index, adjust the first dimension
                    if client_param.shape[0] < self.global_model[param_key].shape[0]:
                        padded_param = torch.zeros_like(self.global_model[param_key])
                        padded_param[:client_param.shape[0], :] = client_param
                        self.global_model[param_key] += padded_param * weight
                    else:
                        self.global_model[param_key] += client_param * weight
            
        # # Initialize global LoRA parameters to zero
        # self.global_model = {k: torch.zeros_like(v) for k, v in self.uploaded_models[0].items()}
        # # print(f'self.global_model before aggregation: {self.global_model}')
        
        # # Aggregate LoRA parameters from each client model
        # for weight, client_model in zip(self.uploaded_weights, self.uploaded_models):
        #     self.add_parameters(weight, client_model)
        # # print(f'self.global_model after aggregation: {self.global_model}')    
            
    def add_parameters(self, weight, client_model):
        for param_key in self.global_model.keys():
            self.global_model[param_key] += client_model[param_key].clone() * weight
    
    def test_metrics_tfl(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        # for c in self.clients[0]:
            
        c = self.clients[0]    
        
        ct, ns, auc = c.test_metrics()
        tot_correct.append(ct*1.0)
        tot_auc.append(auc*ns)
        num_samples.append(ns)
            
            
        print(f'ct, ns, auc: {ct}, {ns}, {auc}')

        # ids = [c.id for c in self.clients]
        
        ids = [c.id]
        
        print(f'ids: {ids}')

        return ids, num_samples, tot_correct, tot_auc
    
    def evaluate(self, acc=None, loss=None):
        
        if not self.pfl:
            stats = self.test_metrics_tfl()
            
            print(f'stats[1]: {stats[1]}')
            print(f'stats[2]: {stats[2]}')
            print(f'stats[3]: {stats[3]}')

            test_acc = sum(stats[2])*1.0 / len(stats[2])
            test_auc = sum(stats[3])*1.0 / len(stats[3])

            accs = 0
            aucs = 0
            
            
        else:
            stats = self.test_metrics()
        
            print(f'stats[1]: {stats[1]}')
            print(f'stats[2]: {stats[2]}')
            print(f'stats[3]: {stats[3]}')

            test_acc = sum(stats[2])*1.0 / len(stats[2])
            test_auc = sum(stats[3])*1.0 / len(stats[3])

            accs = statistics.stdev(stats[2])
            aucs = statistics.stdev(stats[3])
        
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(accs))
        print("Std Test AUC: {:.4f}".format(aucs))
    
    def send_models(self):
        # Instead of sending the whole model, only send LoRA layers
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            # Prepare LoRA parameters truncated for each client's rank
            truncated_lora_params = {}
            for key, value in self.global_model.items():
                # Check if the index of the parameter is even or odd to decide which dimension to truncate
                param_index = int(key.split('_')[-1])  # Extracting the index from the parameter's name like 'lora_param_0'
                if param_index % 2 == 0:
                    # For even indices, the second dimension should be truncated
                    if value.shape[1] > client.lora_params['rank']:
                        truncated_lora_params[key] = value[:, :client.lora_params['rank']].clone()
                    else:
                        truncated_lora_params[key] = value.clone()
                else:
                    # For odd indices, the first dimension should be truncated
                    if value.shape[0] > client.lora_params['rank']:
                        truncated_lora_params[key] = value[:client.lora_params['rank'], ...].clone()
                    else:
                        truncated_lora_params[key] = value.clone()
                    
            # Debug output to check the dimensions
            # print(f"Sending to client {client.id}, Expected Rank: {client.lora_params['rank']}, Param Shapes: {[p.shape for p in truncated_lora_params.values()]}")

            client.set_parameters(truncated_lora_params)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            
            # global_lora_params = self.global_model
            
            # client.set_parameters(global_lora_params)

            # client.send_time_cost['num_rounds'] += 1
            # client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            
    def receive_models(self):
        # Receive only the LoRA layers from each client

        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.clip_model_object.get_lora_state_dict())
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples  
            
        
        # print(f'self.uploaded_weights: {self.uploaded_weights}')
        # for element in self.uploaded_weights:
        #     print(type(element))
            
        # print(f'self.uploaded_models[0]: {self.uploaded_models[0]}')
        # print(f'self.uploaded_models[1]: {self.uploaded_models[1]}')
        
        # print(f'self.uploaded_models length: {len(self.uploaded_models)}')
#         for key, tensor in self.uploaded_models[0].items():
#             print(f"Shape of '{key}': {tensor.shape}")
            
#         for key, tensor in self.uploaded_models[1].items():
#             print(f"Shape of '{key}': {tensor.shape}")
            
        
    
    # ------------------------------------------------------------
    
    
    
    