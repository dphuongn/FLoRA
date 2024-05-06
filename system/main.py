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

#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from flcore.servers.serverlora import FLora
from flcore.servers.serveraa import FedAa
from flcore.servers.serverfft import FedFft
from flcore.servers.serverlc import FedLc
from flcore.servers.servervmlc import FedVmLc
from flcore.servers.serverprompt import FedPrompt
from flcore.servers.serverca import FedCa
from flcore.servers.serverdylora import FedDyLora


from flcore.trainmodel.clip_model import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

from dataset_config import get_class_names

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
# torch.manual_seed(0)


# model_id="openai/clip-vit-base-patch32"
# home_dir=os.path.expanduser('~')


def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        # added-------------------------------------
        
        if args.model == "vit-b-32":
            args.model_id = "openai/clip-vit-base-patch32"
        elif args.model == "vit-b-16":
            args.model_id = "openai/clip-vit-base-patch16"
        elif args.model == "vit-l-14":
            args.model_id = "openai/clip-vit-large-patch14"
        elif args.model == "vit-l-14-336":
            args.model_id = "openai/clip-vit-large-patch14-336"
        else:
            raise NotImplementedError
        
        # if model_str == "lora":
        #     args.model = CLIPModelWithLoRA(model_id=args.model_id, home_dir=args.home_dir, lora_params=lora_params).to(args.device)
            
        # elif model_str == "dynamic_lora":
        #     args.model = CLIPModelWithDynamicLoRA(model_id=args.model_id, home_dir=args.home_dir, lora_params=lora_params).to(args.device)
            
        # elif model_str == "aa":
        #     args.model = CLIPModelWithAttentionAdapter(model_id=args.model_id, home_dir=args.home_dir, aa_params=aa_params).to(args.device)

        # elif model_str == "fft":
        #     args.model = CLIPModelFFT(model_id=args.model_id, home_dir=args.home_dir).to(args.device)
            
        # elif model_str == "lc":
        #     args.model = CLIPModelWithLinearClassifier(model_id=args.model_id, home_dir=args.home_dir, num_classes=len(args.class_names), 
        #                                                 dataset=args.dataset, class_names=args.class_names, device=args.device).to(args.device)

        # elif model_str == "vmlc":
        #     args.model = CLIPModelWithVisionModelLinearClassifier(model_id=args.model_id, home_dir=args.home_dir, num_classes=len(args.class_names),
        #                                                 dataset=args.dataset, class_names=args.class_names, device=args.device).to(args.device)
            
        # elif model_str == "prompt":
        #     args.model = CLIPModelWithPrompt(model_id=args.model_id, home_dir=args.home_dir, class_names=args.class_names, device=args.device).to(args.device)
            
        # elif model_str == "ca":
        #     args.model = CLIPModelWithClipAdapter(model_id=args.model_id, home_dir=args.home_dir, ca_params=ca_params).to(args.device)
        # #-------------------------------------------    

        # else:
        #     raise NotImplementedError

        # print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)
        
        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)
            print(f'hello')
            print(f'args.model: {args.model}')
            print(f'args base model: {args.model.base}')
            print(f'args base model parameters: {args.model.base.parameters()}')
            print(f'args head model: {args.model.head}')
            print(f'args head model parameters: {args.model.head.parameters()}')
            
            break
            
            
        # added---------------------------
        elif args.algorithm == "flora":
            # args.lora = args.model.model.get_lora_state_dict()
            server = FLora(args, i)
            
        elif args.algorithm == "fdylora":
            server = FedDyLora(args, i)
            
        elif args.algorithm == "fedaa":
            server = FedAa(args, i)
            
        elif args.algorithm == "fedfft":
            server = FedFft(args, i)
            
        elif args.algorithm == "fedlc":
            server = FedLc(args, i)
        
        elif args.algorithm == "fedvmlc":
            server = FedVmLc(args, i)
            
        elif args.algorithm == "fedprompt":
            server = FedPrompt(args, i)
            
        elif args.algorithm == "fedca":
            server = FedCa(args, i)
        #---------------------------------

            
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory/1e9)
        
        
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="vit-b-32")
    parser.add_argument('-lbs', "--batch_size", type=int, default=512)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.00005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # MOON
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedAvgDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)
    
    # added ==========================================================================
    parser.add_argument('-tr_d_f', "--train_data_fraction", type=float, default=1.0)
    parser.add_argument('-te_d_f', "--test_data_fraction", type=float, default=1.0)
    parser.add_argument('-sd', "--seed", type=int, default=0, help="Random seed")
    parser.add_argument('-pfl', "--personalized_fl", action='store_true', help="Enable Personalized Federated Learning")
    
    # parser.add_argument('--model_id', type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument('--home_dir', type=str, default=os.path.expanduser('~'))
    
    parser.add_argument('-uw', '--uniform_weight', action='store_true', help="Enable uniform weights")
    parser.add_argument('-norm_rank', '--normalized_rank',  action='store_true', help="Enable normalized LoRA ranks for fdylora")
    parser.add_argument('-norm_alpha', '--normalized_alpha',  action='store_true', help="Enable normalized LoRA alphas for fdylora")
    parser.add_argument('-rand_rank', '--random_rank', action='store_true', help="Enable LoRA random rank")
    parser.add_argument('-rand_alpha', '--random_alpha', action='store_true', help="Enable LoRA random alpha")
    parser.add_argument('-fi_rank', '--fixed_rank', action='store_true', help="Enable LoRA fixed rank")
    parser.add_argument('-fi_alpha', '--fixed_alpha', action='store_true', help="Enable LoRA fixed alpha")
    
    
    # Add these lines in the section where you're defining arguments (in parser.add_argument() calls)
    parser.add_argument('--lora_rank', type=int, default=2, help="LoRA rank")
    parser.add_argument('--lora_rank_min', type=int, default=0, help="LoRA rank min")
    parser.add_argument('--lora_rank_max', type=int, default=32, help="LoRA rank max")
    parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha")
    parser.add_argument('--lora_alpha_min', type=int, default=0, help="LoRA alpha min")
    parser.add_argument('--lora_alpha_max', type=int, default=64, help="LoRA alpha max")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument('--lora_query_text', action='store_true', help="LoRA apply to query text")
    parser.add_argument('--lora_key_text', action='store_true', help="LoRA apply to key text")
    parser.add_argument('--lora_value_text', action='store_true', help="LoRA apply to value text")
    parser.add_argument('--lora_projection_text', action='store_true', help="LoRA apply to projection text")
    parser.add_argument('--lora_mlp_text', action='store_true', help="LoRA apply to MLP text")
    parser.add_argument('--lora_head_text', action='store_true', help="LoRA apply to head text")
    parser.add_argument('--lora_query_vision', action='store_true', help="LoRA apply to query vision")
    parser.add_argument('--lora_key_vision', action='store_true', help="LoRA apply to key vision")
    parser.add_argument('--lora_value_vision', action='store_true', help="LoRA apply to value vision")
    parser.add_argument('--lora_projection_vision', action='store_true', help="LoRA apply to projection vision")
    parser.add_argument('--lora_mlp_vision', action='store_true', help="LoRA apply to MLP vision")
    parser.add_argument('--lora_head_vision', action='store_true', help="LoRA apply to head vision")
    
    
    parser.add_argument('--aa_bottleneck_reduction', type=int, default=1, help="Attention Adapter bottleneck reduction")
    parser.add_argument('--aa_text', action='store_true', help="Attention Adapter apply to text")
    parser.add_argument('--aa_vision', action='store_true', help="Attention Adapter apply to vision")
    
    
    parser.add_argument('--ca_bottleneck_reduction', type=int, default=4, help="Clip Adapter bottleneck reduction")
    parser.add_argument('--ca_ratio', type=float, default=0.2, help="Clip Adapter ratio")
    parser.add_argument('--ca_text', action='store_true', help="Clip Adapter apply to text")
    parser.add_argument('--ca_vision', action='store_true', help="Clip Adapter apply to vision")

    


    args = parser.parse_args()
    
    args.class_names = get_class_names(args.dataset)
    args.num_classes = len(args.class_names)    
    print(f'args.class_names: {args.class_names}')
    print(f'args.num_classes: {args.num_classes}')
    
    args.aa_params = {
        'aa_bottleneck_reduction': args.aa_bottleneck_reduction,
        'aa_text': args.aa_text,
        'aa_vision': args.aa_vision,
    }
    
    
    args.lora_params = {
        'rank': args.lora_rank,
        'rank_min': args.lora_rank_min,
        'rank_max': args.lora_rank_max,
        'alpha': args.lora_alpha,
        'alpha_min': args.lora_alpha_min,
        'alpha_max': args.lora_alpha_max,
        'lora_dropout': args.lora_dropout,
        'lora_query_text': args.lora_query_text,
        'lora_key_text': args.lora_key_text,
        'lora_value_text': args.lora_value_text,
        'lora_projection_text': args.lora_projection_text,
        'lora_mlp_text': args.lora_mlp_text,
        'lora_head_text': args.lora_head_text,
        'lora_query_vision': args.lora_query_vision,
        'lora_key_vision': args.lora_key_vision,
        'lora_value_vision': args.lora_value_vision,
        'lora_projection_vision': args.lora_projection_vision,
        'lora_mlp_vision': args.lora_mlp_vision,
        'lora_head_vision': args.lora_head_vision,
    }


    args.ca_params = {
        'bottleneck_reduction': args.ca_bottleneck_reduction,
        'ratio': args.ca_ratio,
        'aa_text': args.ca_text,
        'aa_vision': args.ca_vision,
    }
    
    
    # Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    
    print("Personalized Federated Learning: {}".format(args.personalized_fl))
    print("Uniform Weight: {}".format(args.uniform_weight))
    print("Normalized Rank: {}".format(args.normalized_rank))
    print("Normalized Alpha: {}".format(args.normalized_alpha))
    
    print("lora_rank: {}".format(args.lora_rank))
    print("lora_rank_min: {}".format(args.lora_rank_min))
    print("lora_rank_max: {}".format(args.lora_rank_max))
    print("lora_alpha: {}".format(args.lora_alpha))
    print("lora_alpha_min: {}".format(args.lora_alpha_min))
    print("lora_alpha_max: {}".format(args.lora_alpha_max))
    print("lora_projection_text: {}".format(args.lora_projection_text))
    print("lora_projection_vision: {}".format(args.lora_projection_vision))
    
    print("aa_bottleneck_reduction: {}".format(args.aa_bottleneck_reduction))
    print("aa_text: {}".format(args.aa_text))
    print("aa_vision: {}".format(args.aa_vision))
    
    print("ca_bottleneck_reduction: {}".format(args.ca_bottleneck_reduction))
    print("ca_text: {}".format(args.ca_text))
    print("ca_vision: {}".format(args.ca_vision))
    

    print("Seed: {}".format(args.seed))
    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch_new))
    print("=" * 50)

    run(args)
