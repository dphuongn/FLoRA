# Introduction

This is the implementation of our paper FLoRA: Enhancing Vision-Language Models with Parameter-Efficient Federated Learning. 


## Environments
With the installed [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh), we can run this platform in a conda virtual environment called *flora*. 
```
conda env create -f env_latest.yaml # for Linux or Windows with CUDA >= 12.1
conda activate flora
```

## Generating datasets

We provide **16** popular datasets: **Fashion-MNIST (F-MNIST)**, **CIFAR-10**, **CIFAR-100**, **Tiny-ImageNet (TINY)**, **Oxford-IIIT Pet (OxfordPets)**, **Oxford 102 Flower (Flowers102)**, **FGVC-Aircraft (Aircraft)**, **Stanford Cars (Cars)**, **Describable Textures Dataset (DTD)**, **EuroSAT**, **FER2013**, **Caltech101**, **Food101**, **Country211**, **SUN397**, and **Rendered SST2 (R-SST2)** and they can be easy split into **IID** and **non-IID** version. For **non-IID**, we have practical setting (with hyperpameter for Dirichlet distribution $\beta$) and pathological setting (few-shot scenario). 

### Examples for **CIFAR-10**
- Total 10 clients, iid and balance scenario
    ```
    cd ./dataset
    python generate_cifar10.py 10 iid balance - - - - 
    ```

- Total 10 clients, practical noniid scenario with $\beta = 0.1$ 
    ```
    cd ./dataset
    python generate_cifar10.py 10 noniid - dir 0.1 - - 
    ```

- Total 10 clients, iid and balance scenario scenario with 1 shot for each client
    ```
    cd ./dataset
    python generate_cifar10.py 10 iid - - - fs 1  
    ```

- Total 5 clients, pathological noniid scenario with 4 shots for each client
    ```
    cd ./dataset
    python generate_cifar10.py 5 noniid - pat - fs 4 
    ```



## Training and Evaluation

After generating and partitioning dataset for clients, we can run the training and evaluation. All codes corresponding to **FLoRA** and other baselines: **FedFFT**, **FedLC**, **FedVM-LC**, and **FedAA** are stored in `./script`. Different folder corresponds with that specific dataset.

### Examples for **FedLC** on **OxfordPets** with **Practical non-IID** scenario
```
cd ./scripts/oxfordpets
bash fedlc.sh
```

### Examples for **FLoRA** on **Caltech101** with **IID few-shot** scenario
```
cd ./scripts/caltech101
bash flora.sh
```

## Parameters

| Parameter | Description |
| --------- | ----------- |
|`data`     | Dataset to use. Options: `fmnist`, `cifar10`, `cifar100`,`tiny`, `pets`, `flowers`, `aircraft`, `cars`, `dtd`, `eurosat`, `fer2013`, `caltech101`, `food101`, `country211`, `sun397`, and `rsst2`.|          
| `m`       | The base model. Options: `vit-b-32`, `vit-b-16`, `vit-l-14`, and `vit-l-14-336` (default: `vit-b-32`).|
| `alg`     | The training algorithm. Options: `fedfft`, `fedlc`, `fedvmlc`, `fedaa`, and `flora`.|
| `gr`      | Number of communication rounds. |
| `jr`      | Ratio of participating clients per round (default: `1`). |
| `did`     | GPU device ID (default: `0`). |
| `nc`      | Number of clients. |
| `lbs`     | Batch size. |
| `lora_rank`               | The LoRA rank for **FLoRA**.|
| `lora_alpha`              | The LoRA scaling factor for **FLoRA**.|
| `lora_projection_text`    | LoRA apply to projection text for **FLoRA**.|
| `lora_projection_vision`  | LoRA apply to projection vision for **FLoRA**.|
| `sd`      | The initial seed. |


Feel free to change parameters to your desired experiments. If you use the same setting as our papers, you can simply adopt the hyperparameters reported in our paper.

# Acknowledgement

This code is heavily inspired from the popoular federated learning project [PFLlib](https://github.com/TsingZ0/PFLlib). Thank you for their wonderful work!

# Citation

- FLoRA
    ```bibtex
    @article{nguyen2025flora,
        title={FLoRA: Enhancing Vision-Language Models with Parameter-Efficient Federated Learning},
        author={Nguyen, Duy Phuong and Muñoz, J. Pablo and Jannesari, Ali},
        journal = {arXiv preprint arXiv:2404.15182},
        year = {2024},
    }
