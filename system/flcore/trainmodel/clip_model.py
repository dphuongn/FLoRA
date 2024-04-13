import torch
from transformers import CLIPProcessor, CLIPModel
from functools import partial
import os
import copy
import torch.nn.functional as F

from utils.data_utils import return_zeroshot_weight


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.W_a = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.W_b = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.W_a @ self.W_b)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

    

class CLIPModelWithLoRA(torch.nn.Module):
    def __init__(self, model_id, home_dir, lora_params):
        """
        Initialize the CLIP model with LoRA layers.
        
        Args:
            model_id (str): Identifier for the pre-trained CLIP model.
            home_dir (str): Directory path for model and processor caching.
            lora_params (dict): Parameters for configuring the LoRA layers.
        """
        super().__init__()
        self.model_id = model_id
        self.home_dir = home_dir
        self.lora_params = lora_params
        self.model = CLIPModel.from_pretrained(self.model_id, cache_dir=f"{self.home_dir}/models")
        self.processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=f"{self.home_dir}/models")
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        self._apply_lora()
        
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def calculate_model_size(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        return size_all_mb
    
    def calculate_state_dict_size(self, state_dict):
        total_size = 0
        for name, tensor in state_dict.items():
            total_size += tensor.nelement() * tensor.element_size()

        size_all_mb = total_size / 1024 ** 2  # Convert to megabytes
        return size_all_mb

        
    def _apply_lora(self):
        """
        Apply LoRA modifications to the CLIP model. This method initializes
        LoRA layers and replaces the corresponding layers in the CLIP model.
        """
        assign_lora = partial(
            LinearWithLoRA,
            rank=self.lora_params['rank'],
            alpha=self.lora_params['alpha']
        )
        
        # Initialize a dictionary to keep track of the LoRA layers
        self.lora_layers = {}
        
        # Apply LoRA modifications to the text and vision models as per the parameters
        for layer in self.model.text_model.encoder.layers:
            if self.lora_params['lora_query_text']:
                layer.self_attn.q_proj = assign_lora(layer.self_attn.q_proj)
            if self.lora_params['lora_key_text']:
                layer.self_attn.k_proj = assign_lora(layer.self_attn.k_proj)
            if self.lora_params['lora_value_text']:
                layer.self_attn.v_proj = assign_lora(layer.self_attn.v_proj)
            if self.lora_params['lora_projection_text']:
                layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
            if self.lora_params['lora_mlp_text']:
                layer.mlp.fc1 = assign_lora(layer.mlp.fc1)
                layer.mlp.fc2 = assign_lora(layer.mlp.fc2)

        if self.lora_params['lora_head_text']:
            self.model.text_projection = assign_lora(self.model.text_projection)


        for layer in self.model.vision_model.encoder.layers:
            if self.lora_params['lora_query_vision']:
                layer.self_attn.q_proj = assign_lora(layer.self_attn.q_proj)
            if self.lora_params['lora_key_vision']:
                layer.self_attn.k_proj = assign_lora(layer.self_attn.k_proj)
            if self.lora_params['lora_value_vision']:
                layer.self_attn.v_proj = assign_lora(layer.self_attn.v_proj)
            if self.lora_params['lora_projection_vision']:
                layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
            if self.lora_params['lora_mlp_vision']:
                layer.mlp.fc1 = assign_lora(layer.mlp.fc1)
                layer.mlp.fc2 = assign_lora(layer.mlp.fc2)

        if self.lora_params['lora_head_vision']:
            self.model.visual_projection = assign_lora(self.model.visual_projection)
            
            
    def _find_lora_layers(self, module, lora_params):
        """
        Recursively find all LoRA layers in the model.

        Args:
            module (torch.nn.Module): The module (or sub-module) to search within.
            lora_params (list): A list to append the parameters of LoRA layers to.
        """
        for child in module.children():
            if isinstance(child, LinearWithLoRA):
                # Assuming LoRALayer is a component of LinearWithLoRA
                lora_params.extend(list(child.lora.parameters()))
            elif isinstance(child, LoRALayer):
                # Directly collecting parameters from LoRALayer
                lora_params.extend(list(child.parameters()))
            else:
                self._find_lora_layers(child, lora_params)

    def get_lora_parameters(self):
        """
        Retrieve all parameters from the LoRA layers in the model.

        Returns:
            list: A list of parameters from all the LoRA layers.
        """
        lora_params = []
        self._find_lora_layers(self.model, lora_params)
        return lora_params
        
    def save_lora_state_dict(self, directory, filename):
        """
        Save the state dictionary of the LoRA layers to a specified file in a given directory.

        Args:
            directory (str): The directory where the state dict file will be saved.
            filename (str): The name of the file to save the state dict.
        """
        lora_params = self.get_lora_parameters()
        state_dict = {f'lora_param_{i}': param.data for i, param in enumerate(lora_params)}

        # Ensure the directory exists, create if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, filename)
        torch.save(state_dict, file_path)
        
    
    def get_lora_state_dict(self):
        """
        Retrieve the state dictionary of the LoRA layers.

        Returns:
            dict: A state dictionary containing the parameters of the LoRA layers.
        """
        lora_params = self.get_lora_parameters()
        state_dict = {f'lora_param_{i}': param.data for i, param in enumerate(lora_params)}
        return state_dict
    
    def set_lora_state_dict(self, state_dict):
        """
        Set the parameters of the LoRA layers from a state dictionary.

        Args:
            state_dict (dict): A state dictionary containing parameters for the LoRA layers.
        """
        lora_params = self.get_lora_parameters()
        for i, param in enumerate(lora_params):
            param_key = f'lora_param_{i}'
            if param_key in state_dict:
                param.data.copy_(state_dict[param_key])
            
    def update_lora_from_state_dict(self, state_dict):
        """
        Update the parameters of the LoRA layers from a state dictionary.

        Args:
            state_dict (dict): A state dictionary containing the parameters of the LoRA layers.
        """
        lora_params = self.get_lora_parameters()

        for i, param in enumerate(lora_params):
            param_key = f'lora_param_{i}'
            if param_key in state_dict:
                param.data.copy_(state_dict[param_key])
            else:
                raise KeyError(f"Parameter key {param_key} not found in the provided state_dict.")  
                
    def load_lora_state_dict(self, file_path):
        """
        Load LoRA parameters from a saved state dictionary file into the model.

        This method updates the LoRA layers in the model with parameters loaded from a file.

        Args:
            file_path (str): The path to the file containing the saved state dictionary.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            KeyError: If a parameter key in the state dictionary does not match any in the model.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        # Load the state dictionary from the file
        state_dict = torch.load(file_path)

        # Update the model's LoRA parameters with the loaded state dictionary
        self.set_lora_state_dict(state_dict)

                
    def print_dict_shapes(self, dictionary):
        """
        Print the shapes of tensors stored in a dictionary.

        This function iterates over each key-value pair in the dictionary.
        It assumes that each value is a tensor and prints the shape of each tensor
        along with its corresponding key.

        Args:
            dictionary (dict): A dictionary where each value is expected to be a tensor.
        """
        for key, tensor in dictionary.items():
            print(f"Shape of '{key}': {tensor.shape}")
            
    def print_dict_values(self, dictionary):
        """
        Print the values of tensors stored in a dictionary.

        This function iterates over each key-value pair in the dictionary.
        It assumes that each value is a tensor and prints the actual values of each tensor
        along with its corresponding key.

        Note: Be cautious when using this method with large tensors, as printing 
        large amounts of data can be time-consuming and may clutter your output.

        Args:
            dictionary (dict): A dictionary where each value is expected to be a tensor.
        """
        for key, tensor in dictionary.items():
            print(f"{key}:\n{tensor}")
            
    def compare_lora_dictionaries(self, dict1, dict2, tolerance=1e-6):
        """
        Compare two dictionaries containing LoRA parameters.

        Args:
            dict1 (dict): The first dictionary of LoRA parameters.
            dict2 (dict): The second dictionary of LoRA parameters.
            tolerance (float): Tolerance level for comparing floating point values.

        Returns:
            bool: True if the dictionaries are the same within the given tolerance, False otherwise.
        """
        
        if dict1.keys() != dict2.keys():
            return False
        
        for key in dict1:
            if key not in dict2 or not torch.allclose(dict1[key], dict2[key], atol=tolerance):
                return False

        return True
    

class CLIPModelWithAttentionAdapter(torch.nn.Module):
    def __init__(self, model_id, home_dir, aa_params):
        """
        Initialize the CLIP model with Attention Adapter from FedCLIP.
        
        Args:
            model_id (str): Identifier for the pre-trained CLIP model.
            home_dir (str): Directory path for model and processor caching.
            aa_params (dict): Parameters for configuring the Attention Adapter layers.
        """
        super().__init__()
        self.model_id = model_id
        self.home_dir = home_dir
        self.aa_params = aa_params
        self.model = CLIPModel.from_pretrained(self.model_id, cache_dir=f"{self.home_dir}/models")
        self.processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=f"{self.home_dir}/models")
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Create and initialize the Attention Adapter
        self.aa = self._make_attention_adapter()
        
    def _make_attention_adapter(self):
        """
        Create the Attention Adapter layers based on the provided parameters.
        """
        in_dim = self.model.visual_projection.out_features
        bottleneck_dim = in_dim // self.aa_params['aa_bottleneck_reduction']
        out_dim = in_dim

        adapter = torch.nn.Sequential(
            torch.nn.Linear(in_dim, bottleneck_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(bottleneck_dim, out_dim),
            torch.nn.Softmax(dim=1),
        )

        # Initialize the adapter layers
        with torch.no_grad():
            # Initialize the first Linear layer to a near-identity matrix
            torch.nn.init.eye_(adapter[0].weight)
            adapter[0].bias.fill_(0.0)

            # Initialize the second Linear layer to a near-identity matrix
            torch.nn.init.eye_(adapter[2].weight)
            adapter[2].bias.fill_(0.0)


        return adapter
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def calculate_model_size(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        return size_all_mb
    
class CLIPModelFFT(torch.nn.Module):
    def __init__(self, model_id, home_dir):
        """
        Initialize the CLIP model.
        
        Args:
            model_id (str): Identifier for the pre-trained CLIP model.
            home_dir (str): Directory path for model and processor caching.
        """
        super().__init__()
        self.model_id = model_id
        self.home_dir = home_dir
        self.model = CLIPModel.from_pretrained(self.model_id, cache_dir=f"{self.home_dir}/models")
        self.processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=f"{self.home_dir}/models")
        

class SingleLayerNN(torch.nn.Module):
    def __init__(self, input_size, num_classes, normalize=False, weights=None, biases=None):
        super(SingleLayerNN, self).__init__()
        self.fc = torch.nn.Linear(input_size, num_classes)
        self.normalize = normalize
        
        # self.fc.weight.detach().zero_()
        # self.fc.bias.detach().zero_()
        
        # Initialize weights and biases if provided
        if weights is not None:
            self.fc.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.fc.bias = torch.nn.Parameter(biases.clone())
        else:
            self.fc.bias = torch.nn.Parameter(torch.zeros_like(self.fc.bias))
        
        # print(f'weight shape: {self.fc.weight.data.shape}')
        # print(f'biases shape: {self.fc.bias.data.shape}')

    def forward(self, x):
        if self.normalize:
            x = x / x.norm(dim=-1, keepdim=True)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        
        return logits, probas

class CLIPModelWithLinearClassifier(torch.nn.Module):
    def __init__(self, model_id, home_dir, num_classes, dataset, class_names, device):
        """
        Initialize the CLIP model with Linear Probing.
        
        Args:
            model_id (str): Identifier for the pre-trained CLIP model.
            home_dir (str): Directory path for model and processor caching.
        """
        super().__init__()
        self.model_id = model_id
        self.home_dir = home_dir
        self.model = CLIPModel.from_pretrained(self.model_id, cache_dir=f"{self.home_dir}/models")
        self.processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=f"{self.home_dir}/models")
        
        self.num_classes = num_classes
        self.dataset = dataset
        self.class_names = class_names
        self.device = device
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Create and initialize the Linear Classifier
        self.lc = self._make_linear_classifier()
        
        
    def _make_linear_classifier(self):
        """
        Creates and returns an instance of the SingleLayerNN linear classifier.
        """
        # Assuming using the output features of the image encoder
        input_size = self.model.visual_projection.out_features
        # zeroshot_weights = self._generate_zeroshot_weights(self.class_names, self.templates, input_size)
        
        zeroshot_weights = return_zeroshot_weight(self.dataset, self.model, self.processor, self.class_names, self.device)
        zeroshot_weights = zeroshot_weights.transpose(0, 1)
        
        return SingleLayerNN(input_size, self.num_classes, normalize=True, weights=zeroshot_weights)
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def calculate_model_size(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        return size_all_mb


class CLIPModelWithVisionModelLinearClassifier(torch.nn.Module):
    def __init__(self, model_id, home_dir, num_classes, dataset, class_names, device):
        """
        Initialize the CLIP model with Image Encoder and Linear Probing.
        
        Args:
            model_id (str): Identifier for the pre-trained CLIP model.
            home_dir (str): Directory path for model and processor caching.
        """
        super().__init__()
        self.model_id = model_id
        self.home_dir = home_dir
        self.model = CLIPModel.from_pretrained(self.model_id, cache_dir=f"{self.home_dir}/models")
        self.processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=f"{self.home_dir}/models")
        
        self.vm = self.model.vision_model
        self.tm = self.model.text_model
        
        self.num_classes = num_classes
        self.dataset = dataset
        self.class_names = class_names
        self.device = device
        
        # Freeze text encoder layers
        for param in self.tm.parameters():
            param.requires_grad = False
        
        # Create and initialize the Linear Classifier
        self.lc = self._make_linear_classifier()
        
    # def _make_linear_classifier(self):
    #     """
    #     Creates and returns an instance of the SingleLayerNN linear classifier.
    #     """
    #     # Assuming using the output features of the image encoder
    #     input_size = self.model.visual_projection.out_features
    #     return SingleLayerNN(input_size, self.num_classes)

    def _make_linear_classifier(self):
        """
        Creates and returns an instance of the SingleLayerNN linear classifier.
        """
        # Assuming using the output features of the image encoder
        input_size = self.model.visual_projection.out_features
        
        zeroshot_weights = return_zeroshot_weight(self.dataset, self.model, self.processor, self.class_names, self.device)
        zeroshot_weights = zeroshot_weights.transpose(0, 1)
        
        return SingleLayerNN(input_size, self.num_classes, normalize=True, weights=zeroshot_weights)
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def calculate_model_size(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        return size_all_mb        
        
class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.text_model.encoder
        self.positional_embedding = clip_model.text_model.embeddings.position_embedding
        self.ln_final = clip_model.text_model.final_layer_norm
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype # not sure

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class PromptLearner(torch.nn.Module):
    def __init__(self, classnames, clip_model, tokenizer, processor, class_token_position):
        super().__init__()
        n_cls = len(classnames)
        # n_ctx = cfg.TRAINER.COOP.N_CTX
        n_ctx = 16
        # ctx_init = cfg.TRAINER.COOP.CTX_INIT
        # ctx_init = ""
        ctx_init = "a photo of a"
        dtype = clip_model.dtype # not sure
        print(f'dtype: {dtype}') 
        # ctx_dim = clip_model.ln_final.weight.shape[0]
        ctx_dim = clip_model.text_model.final_layer_norm.weight.shape[0]
        print(f'ctx_dim: {ctx_dim}')
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            
            print(f'n_ctx length : {n_ctx}')
            
            # prompt = clip.tokenize(ctx_init)
            prompt_raw = tokenizer(ctx_init, padding='max_length', truncation=True,return_tensors="pt")
            
            print(f'prompt_raw: {prompt_raw}')
            
            prompt = prompt_raw["input_ids"]
            
            print(f'prompt shape: {prompt.shape}')
            
            # prompt = prompt.to(self.device)
            with torch.no_grad():
                embedding = clip_model.text_model.embeddings.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
            
        else:

            # ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=self.device)
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            print(f'ctx_vectors shape: {ctx_vectors.shape}')
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        # ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        # print(f'ctx_vectors shape: {ctx_vectors.shape}')
        # nn.init.normal_(ctx_vectors, std=0.02)
        # prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        
        print(f'self.ctx shape: {self.ctx.shape}')

        classnames = [name.replace("_", " ") for name in classnames]
        print(f'classnames: {classnames}')
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        name_lens = [len(tokenizer(name)) for name in classnames]
        print(f'name_lens: {name_lens}')
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print(f'prompts: {prompts}')
        

        
        temp = [tokenizer(p, padding='max_length', truncation=True,return_tensors="pt") for p in prompts]
        
        # print(f'temp: {temp}')
        
        tokenized_prompts = torch.cat([tp["input_ids"] for tp in temp], dim=0)
        
        # print(f'tokenized_prompts: {tokenized_prompts}')

        # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # tokenized_prompts = torch.cat([processor(text=p,
        #                                         images=None,
        #                                         padding='max_length',  
        #                                         max_length=77,       
        #                                         truncation=True,
        #                                         return_tensors='pt') for p in prompts])
        
        with torch.no_grad():
            embedding = clip_model.text_model.embeddings.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        # print(f'self.token_prefix: {self.token_prefix}')
        # print(f'self.token_suffix: {self.token_suffix}')

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        # self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.class_token_position = class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        
        print(f'prefix shape: {prefix.shape}')
        print(f'suffix shape: {suffix.shape}')

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        # print(f'prompts shape: {prompts.shape}')
        
        return prompts

class CLIPModelWithPrompt(torch.nn.Module):
    def __init__(self, model_id, home_dir, class_names, device):
        """
        Initialize the CLIP model with Prompt tuning.
        
        Args:
            model_id (str): Identifier for the pre-trained CLIP model.
            home_dir (str): Directory path for model and processor caching.
        """
        super().__init__()
        self.model_id = model_id
        self.home_dir = home_dir
        self.model = CLIPModel.from_pretrained(self.model_id, cache_dir=f"{self.home_dir}/models")
        self.processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=f"{self.home_dir}/models")
        self.tokenzier = self.processor.tokenizer
        self.vm = self.model.vision_model
        self.tm = self.model.text_model
        self.device = device
        
    
        
        self.class_names = class_names
        self.class_token_position = "end"
        
        self.text_encoder = TextEncoder(self.model)
        
        
        self.prompt_learner = PromptLearner(self.class_names, self.model, self.tokenzier, self.processor, self.class_token_position)
        self.prompts = self.prompt_learner()
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        
        # self.tm2 = TextEncoder(self.model)
        
        self.logit_scale = self.model.state_dict()['logit_scale'].exp()
        
        self.dtype = self.model.dtype # not sure
        
    def get_prompt_state_dict(self):
        """
        Retrieve the state dictionary of the PromptLearner.

        Returns:
            dict: A state dictionary containing the parameters of the PromptLearner.
        """
        return self.prompt_learner.state_dict()
        
    
    def _apply_prompt(self):
        """
        Apply prompt to the CLIP model. This method initializes
        prompt learner in the text encoder of CLIP model.
        """
        
        
    

        
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def calculate_model_size(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        return size_all_mb
    

class CLIPModelWithClipAdapter(torch.nn.Module):
    def __init__(self, model_id, home_dir, ca_params):
        """
        Initialize the CLIP model with Clip Adapter from CLIP-Adapter.
        
        Args:
            model_id (str): Identifier for the pre-trained CLIP model.
            home_dir (str): Directory path for model and processor caching.
            ca_params (dict): Parameters for configuring the Clip Adapter layers.
        """
        super().__init__()
        self.model_id = model_id
        self.home_dir = home_dir
        self.ca_params = ca_params
        self.ratio = self.ca_params['ratio']
        self.model = CLIPModel.from_pretrained(self.model_id, cache_dir=f"{self.home_dir}/models")
        self.processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=f"{self.home_dir}/models")
        
        self.dtype = self.model.dtype 
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Create and initialize the Clip Adapter
        self.ca = self._make_clip_adapter().to(self.model.dtype)
        
        
        
    def _make_clip_adapter(self):
        """
        Creates and returns an instance of the ClipAdapter linear classifier.
        """
        
        in_dim = self.model.visual_projection.out_features
        bottleneck_dim = in_dim // self.ca_params['bottleneck_reduction']
        out_dim = in_dim
        
        adapter = torch.nn.Sequential(
            torch.nn.Linear(in_dim, bottleneck_dim, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(bottleneck_dim, out_dim, bias=False),
            torch.nn.ReLU(inplace=True)
        )        
        
        # Initialize the adapter layers
        with torch.no_grad():
            # Initialize the first Linear layer to a near-identity matrix
            torch.nn.init.eye_(adapter[0].weight)
            # adapter[0].bias.fill_(0.0)

            # Initialize the second Linear layer to a near-identity matrix
            torch.nn.init.eye_(adapter[2].weight)
            # adapter[2].bias.fill_(0.0)

        return adapter
        
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def calculate_model_size(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        return size_all_mb
    
    
    
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)

    # HOME = '/work/LAS/jannesar-lab/dphuong/jupyter'
    HOME='/export/work/yusx/phuong'
    model_id = "openai/clip-vit-base-patch32"
    
    
    class_names = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen', 'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']
    
#     CLIPModelWithPrompt_object = CLIPModelWithPrompt(model_id=model_id, home_dir=HOME, class_names=class_names).to(device)
    
#     prompt_learner = CLIPModelWithPrompt_object.prompt_learner
    
#     prompt_state_dict = CLIPModelWithPrompt_object.get_prompt_state_dict()
    
#     # ctx = prompt_learner.ctx
    
#     # tm2 = CLIPModelWithPrompt_object.tm2
#     # tm = CLIPModelWithPrompt_object.tm
    
#     prompt_learner_params = CLIPModelWithPrompt_object.count_parameters(prompt_learner)  
#     # prompt_learner_params = prompt_learner.ctx.numel()
#     # ctx_params = CLIPModelWithPrompt_object.count_parameters(ctx)  
#     # tm2_params = CLIPModelWithPrompt_object.count_parameters(tm2)
#     # tm_params = CLIPModelWithPrompt_object.count_parameters(tm)
    
#     print(f'prompt_state_dict: {prompt_state_dict}')
#     # print(f'ctx: {ctx}')
# #     print(f'tm2: {tm2}')
# #     print(f'tm: {tm}')
    
#     print(f'prompt_learner_params: {prompt_learner_params}')
#     # print(f'ctx_params: {ctx_params}')
#     # print(f'tm2_params: {tm2_params}')
#     # print(f'tm_params: {tm_params}')
    
    dataset = 'flowers'
    
    CLIPModelWithLinearClassifier_object = CLIPModelWithLinearClassifier(model_id=model_id, home_dir=HOME, num_classes=len(class_names), dataset=dataset, class_names=class_names, device=device).to(device)
    
    
    lc = CLIPModelWithLinearClassifier_object.lc
    
    # Assuming 'lc' is your linear classifier instance
    weights = lc.fc.weight.data
    biases = lc.fc.bias.data

    print("Weights:", weights)
    print("Biases:", biases)
    print(f'shape of weights: {weights.shape}')
    print(f'shape of biases: {biases.shape}')
    
    
    
    
    
    
    lora_params = {
        # 'rank': 8,
        'rank': 2,
        # 'alpha': 16,
        'alpha': 32,
        'lora_dropout': 0.05,

        # 'lora_query_text': True,
        'lora_query_text': False,
        'lora_key_text': False,
        # 'lora_value_text': True,
        'lora_value_text': False,
        # 'lora_projection_text': False,
        'lora_projection_text': True,
        'lora_mlp_text': False,
        'lora_head_text': False,

        'lora_query_vision': False,
        'lora_key_vision': False,
        'lora_value_vision': False,
        'lora_projection_vision': False,
        'lora_mlp_vision': False,
        'lora_head_vision': False,
    }

    
    CLIPModelWithLoRA_object1 = CLIPModelWithLoRA(model_id=model_id, home_dir=HOME, lora_params=lora_params).to(device)
    
    model = CLIPModelWithLoRA_object1.model
    
    lora_params =  CLIPModelWithLoRA_object1.count_parameters(model)
    
    print(f'lora params: {lora_params:,}')
    
    lora_size = CLIPModelWithLoRA_object1.calculate_model_size(model)
    
    print(f'lora size: {lora_size:.3f} MB')
    
#     CLIPModelWithLoRA_object2 = CLIPModelWithLoRA(model_id=model_id, home_dir=HOME, lora_params=lora_params).to(device)
    
#     directory = os.path.join(HOME, 'fed')
#     filename1 = 'first_lora.pt'
#     filename2 = 'second_lora.pt'

#     CLIPModelWithLoRA_object.save_lora_state_dict(directory, filename1)
    
#     lora_1 = CLIPModelWithLoRA_object.get_lora_state_dict()
    
#     CLIPModelWithLoRA_object.print_dict_values(lora_1)
    
    
#     CLIPModelWithLoRA_object.save_lora_state_dict(directory, filename2)
    
#     lora_2 = CLIPModelWithLoRA_object.get_lora_state_dict()
    
#     CLIPModelWithLoRA_object.print_dict_values(lora_2)
    
    
#     file_path1 = os.path.join(directory, filename1)
    
#     CLIPModelWithLoRA_object1.load_lora_state_dict(file_path1)
    
#     lora_1 = CLIPModelWithLoRA_object1.get_lora_state_dict()
    
    # CLIPModelWithLoRA_object.print_dict_values(lora_1)
    
    # lora_1 = copy.deepcopy(lora_1)
    
    # print(f'lora_1: {lora_1}')
    
    
#     file_path2 = os.path.join(directory, filename2)
    
#     CLIPModelWithLoRA_object1.load_lora_state_dict(file_path2)
    
#     lora_2 = CLIPModelWithLoRA_object1.get_lora_state_dict()
    
    # CLIPModelWithLoRA_object.print_dict_values(lora_2)
    
    # lora_2 = copy.deepcopy(lora_2)
    
    # print(f'lora_2: {lora_2}')
    
    
    
    # print(CLIPModelWithLoRA_object1.compare_lora_dictionaries(lora_1, lora_2))
    
    
    
#     aa_params = {
#         'bottleneck_reduction': 1,

#         'aa_text': False,

#         'aa_vision': True,
#     }

#     CLIPModelWithAttentionAdapter_object1 = CLIPModelWithAttentionAdapter(model_id=model_id, 
#                                                                           home_dir=HOME, aa_params=aa_params).to(device)
    
    
#     aa = CLIPModelWithAttentionAdapter_object1.aa
    
#     print(f'aa: {aa}')
    
#     for layer in aa:
#         if isinstance(layer, torch.nn.Linear):
#             print(f"Weights of Linear layer: {layer.weight.data}")
#             print(f"Bias of Linear layer: {layer.bias.data}")
    
#     aa_params = CLIPModelWithAttentionAdapter_object1.count_parameters(aa)
    
#     print(f'aa params: {aa_params:,}')
    
#     aa_size = CLIPModelWithAttentionAdapter_object1.calculate_model_size(aa)
    
#     print(f'aa size: {aa_size:.3f} MB')
    
    
    
    
    
#     num_classes = 102
    
    
#     CLIPModelWithVisionModelLinearClassifier = CLIPModelWithVisionModelLinearClassifier(model_id=model_id, 
#                                                                 home_dir=HOME, num_classes=num_classes).to(device)
    
#     model = CLIPModelWithVisionModelLinearClassifier.model
    
#     vm = CLIPModelWithVisionModelLinearClassifier.vm
    
#     tm = CLIPModelWithVisionModelLinearClassifier.tm
    
#     lc = CLIPModelWithVisionModelLinearClassifier.lc
    
    
    # print(f'lc: {lc}')
    # print(f"Weights of Linear layer: {lc.fc.weight.data}")
    # print(f"Bias of Linear layer: {lc.fc.bias.data}")
    
            
#     lc_params = CLIPModelWithVisionModelLinearClassifier.count_parameters(lc) 
    
#     lc_size = CLIPModelWithVisionModelLinearClassifier.calculate_model_size(lc)
    
#     vm_params = CLIPModelWithVisionModelLinearClassifier.count_parameters(vm) 
    
#     vm_size = CLIPModelWithVisionModelLinearClassifier.calculate_model_size(vm)
    
#     vm_lc_size = lc_size + vm_size
    
#     print(f'lc_params: {lc_params:,}')
    
#     print(f'lc_size: {lc_size:.3f} MB')
    
#     print(f'vm_params: {vm_params:,}')
    
#     print(f'vm_lc_size: {vm_lc_size:.3f} MB')
    
    
    
#     CLIPModelFFT = CLIPModelFFT(model_id=model_id, home_dir=HOME).to(device)
    
#     clip = CLIPModelFFT.model
    
#     clip_size = CLIPModelFFT.calculate_model_size(clip)
    
#     processor = CLIPModelFFT.processor
    
#     vm = CLIPModelFFT.vm
    
#     tm = CLIPModelFFT.tm
    
#     clip_params = CLIPModelFFT.count_parameters(clip)
    
#     vm_params = CLIPModelFFT.count_parameters(vm) 
    
#     tm_params = CLIPModelFFT.count_parameters(tm)
    
#     clip_params_plus = vm_params + tm_params
    
    
    
    
    
#     print(f'clip_params: {clip_params:,}')
    
#     print(f'clip_size: {clip_size:.3f} MB')
    
#     print(f'vm_params: {vm_params:,}')
    
#     print(f'tm_params: {tm_params:,}')
    
#     print(f'clip_params_plus: {clip_params_plus:,}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    