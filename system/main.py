
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

from flcore.servers.serverlora import FedLora
from flcore.servers.serveraa import FedAa
from flcore.servers.serverfft import FedFft
from flcore.servers.serverlc import FedLc
from flcore.servers.servervmlc import FedVmLc
from flcore.servers.serverprompt import FedPrompt
from flcore.servers.serverca import FedCa


from flcore.trainmodel.clip_model import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

from dataset_config import get_class_names

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
# torch.manual_seed(0)


model_id="openai/clip-vit-base-patch32"
home_dir=os.path.expanduser('~')

ca_params = {
    'bottleneck_reduction': 4,
    
    'ratio': 0.6,
    
    'aa_text': False,
    
    'aa_vision': True,
}


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
        if model_str == "lora":
            args.model = CLIPModelWithLoRA(model_id=model_id, home_dir=home_dir, lora_params=lora_params).to(args.device)
            
        elif model_str == "aa":
            args.model = CLIPModelWithAttentionAdapter(model_id=model_id, home_dir=home_dir, aa_params=aa_params).to(args.device)

        elif model_str == "fft":
            args.model = CLIPModelFFT(model_id=model_id, home_dir=home_dir).to(args.device)
            
        elif model_str == "lc":
            args.model = CLIPModelWithLinearClassifier(model_id=model_id, home_dir=home_dir, num_classes=len(args.class_names), 
                                                        dataset=args.dataset, class_names=args.class_names, device=args.device).to(args.device)

        elif model_str == "vmlc":
            args.model = CLIPModelWithVisionModelLinearClassifier(model_id=model_id, home_dir=home_dir, num_classes=len(args.class_names),
                                                        dataset=args.dataset, class_names=args.class_names, device=args.device).to(args.device)
            
        elif model_str == "prompt":
            args.model = CLIPModelWithPrompt(model_id=model_id, home_dir=home_dir, class_names=args.class_names, device=args.device).to(args.device)
            
        elif model_str == "ca":
            args.model = CLIPModelWithClipAdapter(model_id=model_id, home_dir=home_dir, ca_params=ca_params).to(args.device)
        #-------------------------------------------    

        else:
            raise NotImplementedError

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
        elif args.algorithm == "fedlora":
            # args.lora = args.model.model.get_lora_state_dict()
            server = FedLora(args, i)
            
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
    parser.add_argument('-m', "--model", type=str, default="cnn")
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
    parser.add_argument('-fs', "--few_shot", type=bool, default=False)
    
    
    # Add these lines in the section where you're defining arguments (in parser.add_argument() calls)
    parser.add_argument('--lora_rank', type=int, default=2, help="LoRA rank")
    parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument('--lora_query_text', type=bool, default=False, help="LoRA apply to query text")
    parser.add_argument('--lora_key_text', type=bool, default=False, help="LoRA apply to key text")
    parser.add_argument('--lora_value_text', type=bool, default=False, help="LoRA apply to value text")
    parser.add_argument('--lora_projection_text', type=bool, default=False, help="LoRA apply to projection text")
    parser.add_argument('--lora_mlp_text', type=bool, default=False, help="LoRA apply to MLP text")
    parser.add_argument('--lora_head_text', type=bool, default=False, help="LoRA apply to head text")
    parser.add_argument('--lora_query_vision', type=bool, default=False, help="LoRA apply to query vision")
    parser.add_argument('--lora_key_vision', type=bool, default=False, help="LoRA apply to key vision")
    parser.add_argument('--lora_value_vision', type=bool, default=False, help="LoRA apply to value vision")
    parser.add_argument('--lora_projection_vision', type=bool, default=False, help="LoRA apply to projection vision")
    parser.add_argument('--lora_mlp_vision', type=bool, default=False, help="LoRA apply to MLP vision")
    parser.add_argument('--lora_head_vision', type=bool, default=False, help="LoRA apply to head vision")
    
    
    parser.add_argument('--aa_bottleneck_reduction', type=int, default=1, help="Attention Adapter bottleneck reduction")
    parser.add_argument('--aa_text', type=bool, default=False, help="Attention Adapter apply to text")
    parser.add_argument('--aa_vision', type=bool, default=False, help="Attention Adapter apply to vision")

    


    args = parser.parse_args()
    
    
    args.class_names = get_class_names(args.dataset)
    
    print(f'args.class_names: {args.class_names}')
    
    aa_params = {
        'aa_bottleneck_reduction': args.aa_bottleneck_reduction,
        'aa_text': args.aa_text,
        'aa_vision': args.aa_vision,
    }
    
    
    lora_params = {
        'rank': args.lora_rank,
        'alpha': args.lora_alpha,
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

    
    
    # Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

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
