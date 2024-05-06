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

import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data, read_client_data_clip, return_zeroshot_weight, accuracy
from torch.utils.data import Subset

from flcore.trainmodel.clip_model import *


class clientVMLC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.class_names = args.class_names
        
        self.clip_model_object = CLIPModelWithVisionModelLinearClassifier(model_id=args.model_id, home_dir=args.home_dir, num_classes=args.num_classes,
                                                        dataset=args.dataset, class_names=self.class_names, device=args.device).to(args.device)
        
        self.clip_model = self.clip_model_object.model
        
        self.vm = self.clip_model_object.vm
        
        self.lc = self.clip_model_object.lc
        
        self.processor = self.clip_model_object.processor
        
        self.loss = nn.CrossEntropyLoss()
        
        self.train_data_fraction = args.train_data_fraction
        self.test_data_fraction = args.test_data_fraction
        
        self.optimizer = torch.optim.Adam(
            [{'params': [p for p in self.vm.parameters() if p.requires_grad]},
             {'params': [p for p in self.lc.parameters() if p.requires_grad]}],
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.2
        )
        
        
        # print(f"print LoRA parameters before training:")
        # for name, param in self.clip_model.named_parameters():
        #     # Check if the parameter's parent module is a LoRALayer
        #     if 'lora' in name:
        #         print(f"{name}: {param.data}")
        
#         num_param = self.clip_model_object.count_parameters()
#         print("Trained parameters in model: {:,}".format(num_param))
        
#         clip_model_size = self.clip_model_object.calculate_model_size(self.clip_model)
#         print('Size of clip model: {:.3f} MB'.format(clip_model_size))
        
#         lora_state_dict = self.clip_model_object.get_lora_state_dict()
#         lora_state_dict_size = self.clip_model_object.calculate_state_dict_size(lora_state_dict) 
#         print('Size of lora state edict: {:.3f} MB'.format(lora_state_dict_size))

        
        
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data_clip(self.dataset, self.id, self.processor, self.class_names, self.device, is_train=True)
        
        train_subset_size = int(len(train_data) * self.train_data_fraction)
        train_indices = np.random.choice(len(train_data), train_subset_size, replace=False)
        train_subset = Subset(train_data, train_indices)
        
        return DataLoader(train_subset, batch_size, drop_last=False, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data_clip(self.dataset, self.id, self.processor, self.class_names, self.device, is_train=False)
        
        test_subset_size = int(len(test_data) * self.test_data_fraction)
        test_indices = np.random.choice(len(test_data), test_subset_size, replace=False)
        test_subset = Subset(test_data, test_indices)
        
        return DataLoader(test_subset, batch_size, drop_last=False, shuffle=False)

    def train(self):
        trainloader = self.load_train_data()
        # self.clip_model.to(self.device)
        self.clip_model.train()
        self.lc.train()

        # differential privacy
        if self.privacy:
            model_origin = copy.deepcopy(self.clip_model)
            self.clip_model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.clip_model, self.optimizer, trainloader, self.dp_sigma)
        
        start = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            with tqdm(trainloader, total=len(trainloader)) as pbar:  # Initialize pbar here
                for batch in pbar:      

                    images, target, texts = batch
                    
                    target = target.to(self.device)

                    image_features = self.clip_model.get_image_features(images).float()
                    
                    # added LC-----------------------------------------
                    logits, probas = self.lc(image_features)
                    #--------------------------------------------------
        
        
                    # Compute loss
                    loss = self.loss(logits, target)

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    pbar.set_description(f"Epoch {epoch+1}/{self.local_epochs}, Loss: {loss.item():.4f}")

        end = time.time()
        elapsed = end-start
        print(f"Time elapsed {elapsed/60:.2f} min")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += elapsed

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

            for param, param_dp in zip(model_origin.parameters(), self.clip_model.parameters()):
                param.data = param_dp.data.clone()
            self.clip_model = model_origin
            self.optimizer = torch.optim.SGD(self.clip_model.parameters(), lr=self.learning_rate)
            
            
            
    # added ------------------------------------------------------
    
    
    
    def test_metrics(self):
        testloaderfull = self.load_test_data()
        
        # self.clip_modelmodel = self.load_model('model')
        # self.clip_model.to(self.device)
        self.clip_model.eval()
        self.lc.eval()
                
        with torch.no_grad():
            top1_1, top5_1, test_num = 0., 0., 0.

            # for i, (images, target, texts) in enumerate(tqdm(testloaderfull)):
            for i, (images, target, texts) in enumerate(testloaderfull):
                images = images
                target = target.to(self.device)
                texts = texts

                # predict
                image_features = self.clip_model.get_image_features(images)
                
                # added LC-----------------------------------------
                logits, probas = self.lc(image_features)
                #--------------------------------------------------
                
                # image_features /= image_features.norm(dim=-1, keepdim=True)

                # measure accuracy of 1 template
                # zeroshot_weights_1 = return_zeroshot_weight(self.dataset, self.clip_model, self.processor, self.class_names, self.device)
                # logits = 100. * image_features @ zeroshot_weights_1
                # acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                # acc1, acc5 = accuracy(probas, target, topk=(1, 5))
                acc1 = accuracy(logits, target, topk=(1,))
                top1_1 += acc1[0]
                # top5_1 += acc5

                test_num += images.size(0)

        top1_1 = (top1_1 / test_num) * 100
        top5_1 = (top5_1 / test_num) * 100 

        print(f"accuracy of 1 template:")
        print(f"Top-1: {top1_1:.2f}, Top-5: {top5_1:.2f}")
    
        return top1_1, test_num, 0
    
    def set_vm_parameters(self, model):
        self.clip_model_object.set_vm_parameters(model)
        
    def set_lc_parameters(self, model):
        self.clip_model_object.set_lc_parameters(model)
    
    # ------------------------------------------------------------
    
    
    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    HOME = '/work/LAS/jannesar-lab/dphuong/jupyter'
    model_id = "openai/clip-vit-base-patch32"
    
    
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=f"{HOME}/models")
    
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)
    
    
#     client = clientAVGC(, 
#                             id=i, 
#                             train_samples=len(train_data), 
#                             test_samples=len(test_data), 
#                             train_slow=train_slow, 
#                             send_slow=send_slow)
#             self.clients.append(client)
    
    
    