
import numpy as np
import os
import torch
from PIL import Image
from tqdm.auto import tqdm

from torchvision.transforms import ToPILImage

from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader


def read_data(dataset, idx, is_train=True):
    if is_train:
        current_directory = os.getcwd()
        print("Current Working Directory:", current_directory)
        
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data
    

    
class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, device, class_names):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.class_names = class_names
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        # label_name = self.dataset.info.features['label'].names[label]
        label_name = self.class_names[label]
        clip_label = [f"a photo of a {label_name}."]
        
        # Convert to PIL image if not already
        # if not isinstance(image, Image.Image):
        #     image = Image.fromarray(image)
            
        # Convert the tensor back to a PIL Image
        if not isinstance(image, Image.Image):
            image = ToPILImage()(image)
        
        label_token = self.processor(
            text=clip_label,
            images=None,
            padding='max_length',  # Add padding to a specified maximum length
            max_length=77,         # Example max length, adjust as needed
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # or use like this -----------------
        transformed_image = self.processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to(self.device)
        # ----------------------------------
        
        transformed_image = transformed_image.squeeze(0)

        return transformed_image, label, label_token
    
class DIGIT5Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, device, class_names):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.class_names = class_names
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        # label_name = self.dataset.info.features['label'].names[label]
        label_name = self.class_names[label]
        clip_label = [f"a photo of the number {label_name}."]
        
        # Convert to PIL image if not already
        # if not isinstance(image, Image.Image):
        #     image = Image.fromarray(image)
            
        # Convert the tensor back to a PIL Image
        if not isinstance(image, Image.Image):
            image = ToPILImage()(image)
        
        label_token = self.processor(
            text=clip_label,
            images=None,
            padding='max_length',  # Add padding to a specified maximum length
            max_length=77,         # Example max length, adjust as needed
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # or use like this -----------------
        transformed_image = self.processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to(self.device)
        # ----------------------------------
        
        transformed_image = transformed_image.squeeze(0)

        return transformed_image, label, label_token
    
class PETSDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, device, class_names):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.class_names = class_names
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        # label_name = self.dataset.info.features['label'].names[label]
        label_name = self.class_names[label]
        clip_label = [f"a photo of a {label_name}, a type of pet."]
            
        # Convert the tensor back to a PIL Image
        if not isinstance(image, Image.Image):
            image = ToPILImage()(image)
        
        label_token = self.processor(
            text=clip_label,
            images=None,
            padding='max_length',  # Add padding to a specified maximum length
            max_length=77,         # Example max length, adjust as needed
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # or use like this -----------------
        transformed_image = self.processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to(self.device)
        # ----------------------------------
        
        transformed_image = transformed_image.squeeze(0)

        return transformed_image, label, label_token
    
class OXFORDFLOWERSDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, device, class_names):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.class_names = class_names
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        # label_name = self.dataset.info.features['label'].names[label]
        label_name = self.class_names[label]
        clip_label = [f"a photo of a {label_name}, a type of flower."]
            
        # Convert the tensor back to a PIL Image
        if not isinstance(image, Image.Image):
            image = ToPILImage()(image)
        
        label_token = self.processor(
            text=clip_label,
            images=None,
            padding='max_length',  # Add padding to a specified maximum length
            max_length=77,         # Example max length, adjust as needed
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # or use like this -----------------
        transformed_image = self.processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to(self.device)
        # ----------------------------------
        
        transformed_image = transformed_image.squeeze(0)

        return transformed_image, label, label_token
    
class COUNTRY211Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, device, class_names):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.class_names = class_names
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        # label_name = self.dataset.info.features['label'].names[label]
        label_name = self.class_names[label]
        clip_label = [f"a photo i took in {label_name}."]
            
        # Convert the tensor back to a PIL Image
        if not isinstance(image, Image.Image):
            image = ToPILImage()(image)
        
        label_token = self.processor(
            text=clip_label,
            images=None,
            padding='max_length',  # Add padding to a specified maximum length
            max_length=77,         # Example max length, adjust as needed
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # or use like this -----------------
        transformed_image = self.processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to(self.device)
        # ----------------------------------
        
        transformed_image = transformed_image.squeeze(0)

        return transformed_image, label, label_token
    
    
class AIRCRAFTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, device, class_names):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.class_names = class_names
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        # label_name = self.dataset.info.features['label'].names[label]
        label_name = self.class_names[label]
        clip_label = [f"a photo of a {label_name}, a type of aircraft."]
            
        # Convert the tensor back to a PIL Image
        if not isinstance(image, Image.Image):
            image = ToPILImage()(image)
        
        label_token = self.processor(
            text=clip_label,
            images=None,
            padding='max_length',  # Add padding to a specified maximum length
            max_length=77,         # Example max length, adjust as needed
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # or use like this -----------------
        transformed_image = self.processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to(self.device)
        # ----------------------------------
        
        transformed_image = transformed_image.squeeze(0)

        return transformed_image, label, label_token
    
class FOOD101Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, device, class_names):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.class_names = class_names
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        # label_name = self.dataset.info.features['label'].names[label]
        label_name = self.class_names[label]
        clip_label = [f"a photo of {label_name}, a type of food."]
            
        # Convert the tensor back to a PIL Image
        if not isinstance(image, Image.Image):
            image = ToPILImage()(image)
        
        label_token = self.processor(
            text=clip_label,
            images=None,
            padding='max_length',  # Add padding to a specified maximum length
            max_length=77,         # Example max length, adjust as needed
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # or use like this -----------------
        transformed_image = self.processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to(self.device)
        # ----------------------------------
        
        transformed_image = transformed_image.squeeze(0)

        return transformed_image, label, label_token
    
class DTDDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, device, class_names):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.class_names = class_names
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        # label_name = self.dataset.info.features['label'].names[label]
        label_name = self.class_names[label]
        clip_label = [f"A photo of a {label_name} texture."]
            
        # Convert the tensor back to a PIL Image
        if not isinstance(image, Image.Image):
            image = ToPILImage()(image)
        
        label_token = self.processor(
            text=clip_label,
            images=None,
            padding='max_length',  # Add padding to a specified maximum length
            max_length=77,         # Example max length, adjust as needed
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # or use like this -----------------
        transformed_image = self.processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to(self.device)
        # ----------------------------------
        
        transformed_image = transformed_image.squeeze(0)

        return transformed_image, label, label_token

class EUROSATDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, device, class_names):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.class_names = class_names
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        # label_name = self.dataset.info.features['label'].names[label]
        label_name = self.class_names[label]
        clip_label = [f"A centered satellite photo of {label_name}."]
            
        # Convert the tensor back to a PIL Image
        if not isinstance(image, Image.Image):
            image = ToPILImage()(image)
        
        label_token = self.processor(
            text=clip_label,
            images=None,
            padding='max_length',  # Add padding to a specified maximum length
            max_length=77,         # Example max length, adjust as needed
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # or use like this -----------------
        transformed_image = self.processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to(self.device)
        # ----------------------------------
        
        transformed_image = transformed_image.squeeze(0)

        return transformed_image, label, label_token
    
class FER2013Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, device, class_names):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.class_names = class_names
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        # label_name = self.dataset.info.features['label'].names[label]
        label_name = self.class_names[label]
        clip_label = [f"A photo of a {label_name} looking face."]
            
        # Convert the tensor back to a PIL Image
        if not isinstance(image, Image.Image):
            image = ToPILImage()(image)
        
        label_token = self.processor(
            text=clip_label,
            images=None,
            padding='max_length',  # Add padding to a specified maximum length
            max_length=77,         # Example max length, adjust as needed
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # or use like this -----------------
        transformed_image = self.processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to(self.device)
        # ----------------------------------
        
        transformed_image = transformed_image.squeeze(0)

        return transformed_image, label, label_token
    
class RSST2Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, device, class_names):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.class_names = class_names
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        # label_name = self.dataset.info.features['label'].names[label]
        label_name = self.class_names[label]
        clip_label = [f"a {label_name} review of a movie."]
            
        # Convert the tensor back to a PIL Image
        if not isinstance(image, Image.Image):
            image = ToPILImage()(image)
        
        label_token = self.processor(
            text=clip_label,
            images=None,
            padding='max_length',  # Add padding to a specified maximum length
            max_length=77,         # Example max length, adjust as needed
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # or use like this -----------------
        transformed_image = self.processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to(self.device)
        # ----------------------------------
        
        transformed_image = transformed_image.squeeze(0)

        return transformed_image, label, label_token
    
class PCAMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, device, class_names):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.class_names = class_names
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        # label_name = self.dataset.info.features['label'].names[label]
        label_name = self.class_names[label]
        clip_label = [f"This is a photo of {label_name}."]
            
        # Convert the tensor back to a PIL Image
        if not isinstance(image, Image.Image):
            image = ToPILImage()(image)
        
        label_token = self.processor(
            text=clip_label,
            images=None,
            padding='max_length',  # Add padding to a specified maximum length
            max_length=77,         # Example max length, adjust as needed
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # or use like this -----------------
        transformed_image = self.processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to(self.device)
        # ----------------------------------
        
        transformed_image = transformed_image.squeeze(0)

        return transformed_image, label, label_token

    
def read_client_data_clip(dataset, idx, processor, class_names, device, is_train=True):
    
    if is_train:
        
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        
        if 'cifar10' in dataset:
            train_dataset = CIFAR10Dataset(train_data, processor, device, class_names)
            
        elif 'tiny' in dataset:
            train_dataset = CIFAR10Dataset(train_data, processor, device, class_names)
            
        elif dataset == 'cars':
            train_dataset = CIFAR10Dataset(train_data, processor, device, class_names)
        
        elif dataset == 'digit5':
            train_dataset = DIGIT5Dataset(train_data, processor, device, class_names)
            
        elif dataset == 'DomainNet':
            train_dataset = CIFAR10Dataset(train_data, processor, device, class_names) 
        
        elif dataset == 'fmnist':
            train_dataset = CIFAR10Dataset(train_data, processor, device, class_names)
            
        elif dataset == 'pets':
            train_dataset = PETSDataset(train_data, processor, device, class_names)
            
        elif dataset == 'flowers':
            train_dataset = OXFORDFLOWERSDataset(train_data, processor, device, class_names)
            
        elif 'country211' in dataset:
            train_dataset = COUNTRY211Dataset(train_data, processor, device, class_names)
            
        elif dataset == 'aircraft':
            train_dataset = AIRCRAFTDataset(train_data, processor, device, class_names)
            
        elif 'food101' in dataset: 
            train_dataset = FOOD101Dataset(train_data, processor, device, class_names)
            
        elif dataset == 'dtd':
            train_dataset = DTDDataset(train_data, processor, device, class_names)
            
        elif dataset == 'eurosat':
            train_dataset = EUROSATDataset(train_data, processor, device, class_names)
            
        elif dataset == 'fer2013':
            train_dataset = FER2013Dataset(train_data, processor, device, class_names)
            
        elif dataset == 'caltech101':
            train_dataset = CIFAR10Dataset(train_data, processor, device, class_names)
            
        elif 'sun397' in dataset:
            train_dataset = CIFAR10Dataset(train_data, processor, device, class_names)
            
        elif dataset == 'rsst2':
            train_dataset = RSST2Dataset(train_data, processor, device, class_names) 
        
        elif dataset == 'pcam':
            train_dataset = PCAMDataset(train_data, processor, device, class_names) 
            
        
        return train_dataset
        
    else:
        
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
    
        if 'cifar10' in dataset:
            test_dataset = CIFAR10Dataset(test_data, processor, device, class_names)
            
        elif 'tiny' in dataset:
            test_dataset = CIFAR10Dataset(test_data, processor, device, class_names)
            
        elif dataset == 'cars':
            test_dataset = CIFAR10Dataset(test_data, processor, device, class_names)
        
        elif dataset == 'digit5':
            test_dataset = DIGIT5Dataset(test_data, processor, device, class_names)
            
        elif dataset == 'DomainNet':
            test_dataset = CIFAR10Dataset(test_data, processor, device, class_names) 
        
        elif dataset == 'fmnist':
            test_dataset = CIFAR10Dataset(test_data, processor, device, class_names)
            
        elif dataset == 'pets':
            test_dataset = PETSDataset(test_data, processor, device, class_names)
            
        elif dataset == 'flowers':
            test_dataset = OXFORDFLOWERSDataset(test_data, processor, device, class_names)
            
        elif 'country211' in dataset:
            test_dataset = COUNTRY211Dataset(test_data, processor, device, class_names)
        
        elif dataset == 'aircraft':
            test_dataset = AIRCRAFTDataset(test_data, processor, device, class_names)
            
        elif 'food101' in dataset:
            test_dataset = FOOD101Dataset(test_data, processor, device, class_names)
            
        elif dataset == 'dtd':
            test_dataset = DTDDataset(test_data, processor, device, class_names)
            
        elif dataset == 'eurosat':
            test_dataset = EUROSATDataset(test_data, processor, device, class_names)
            
        elif dataset == 'fer2013':
            test_dataset = FER2013Dataset(test_data, processor, device, class_names)
            
        elif dataset == 'caltech101':
            test_dataset = CIFAR10Dataset(test_data, processor, device, class_names)
            
        elif 'sun397' in dataset:
            test_dataset = CIFAR10Dataset(test_data, processor, device, class_names)
            
        elif dataset == 'rsst2':
            test_dataset = RSST2Dataset(test_data, processor, device, class_names) 
            
        elif dataset == 'pcam':
            test_dataset = PCAMDataset(test_data, processor, device, class_names) 
    
        return test_dataset
    

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    return [float(correct[:k].reshape(-1).float().sum().item()) for k in topk]  # Use .item() instead
    
    
    
def zeroshot_classifier(classnames, templates, model, processor, device):
    with torch.no_grad():
        model.to(device)  # Ensure the model is on the correct device
        zeroshot_weights = []
    for classname in tqdm(classnames):
        texts = [template.format(classname) for template in templates] #format with class

        text_tokens = processor(
            text=texts,
            padding=True,
            images=None,
            return_tensors='pt'
        ).to(device)

        class_embeddings = model.get_text_features(**text_tokens) #embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def return_zeroshot_weight(dataset, model, processor, class_names, device):
    
    templates_80 = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]
    templates_7 = '''itap of a {}.
    a bad photo of the {}.
    a origami {}.
    a photo of the large {}.
    a {} in a video game.
    art of the {}.
    a photo of the small {}.'''.split('\n')
    
    if dataset == 'digit5':
        templates_1 = ['a photo of the number {}.']
    
    elif dataset == 'pets':
        templates_1 = ['a photo of a {}, a type of pet.']    
        
    elif dataset == 'flowers':
        templates_1 = ['a photo of a {}, a type of flower.']    
        
    elif 'country211' in dataset:
        templates_1 = ['a photo i took in {}.']
        
    elif dataset == 'aircraft':
        templates_1 = ['a photo of a {}, a type of aircraft.']
        
    elif 'food101' in dataset:
        templates_1 = ['a photo of {}, a type of food.']
        
    elif dataset == 'dtd':
        templates_1 = ['a photo of a {} texture.']
        
    elif dataset == 'eurosat':
        templates_1 = ['a centered satellite photo of {}.']
        
    elif dataset == 'fer2013':
        templates_1 = ['a photo of a {} looking face.']
    
    elif dataset == 'rsst2':
        templates_1 = ['a {} review of a movie.']
    
    elif dataset == "pcam":
        templates_1 = ['this is a photo of {}.']
    
    else: 
        templates_1 = ['a photo of a {}.']
        
    
    zeroshot_weights = zeroshot_classifier(class_names, templates_1, model, processor, device)
    
    return zeroshot_weights
    


def read_client_data(dataset, idx, is_train=True):
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx, is_train)
    elif dataset[:2] == "sh":
        return read_client_data_shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    HOME = '/work/LAS/jannesar-lab/dphuong/jupyter'
    model_id = "openai/clip-vit-base-patch32"
    
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=f"{HOME}/models")
    
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)
    
    train_data = read_client_data_clip('digit5', 1, processor, class_names, device, is_train=True)
    
    print(f'train_data: {train_data}')
    
    train_dataloader = DataLoader(train_data, 5, drop_last=True, shuffle=False)
    
    x = next(iter(train_dataloader))
    print(x[0].shape)
    print(x[1].shape)
    
