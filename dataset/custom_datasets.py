import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder 
from PIL import Image
from transformers import BertTokenizer

from dataset.util import get_dataset
from transformers import ViTImageProcessor

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class My_dataset(Dataset):
    def __init__(self, data: str = "CelebA", root_dir: str='dataset/data/', split: str='train', shuffle: bool = False, ratio: float = 0.95):
        super().__init__()
        self.data_name = data
        self.root_dir = root_dir
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        self.split = split
        self.shuffle = shuffle
        self.ratio = ratio
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        assert self.split in self.split_dict.keys(), "Not Defined Split!"
        self.data_idx, self.attr, self.transform = get_dataset(self.data_name, self.root_dir, self.split, self.split_dict, shuffle=self.shuffle, ratio=self.ratio)
        
        self.attr = torch.LongTensor(self.attr)
        
    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, idx):
        if self.data_name == 'CivilComments':
            raw_text = self.data_idx[idx]
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)

            tokens = self.tokenizer(
                raw_text,
                padding="max_length",
                truncation=True,
                max_length=220,
                return_tensors="pt",
        )
            return_tokens = torch.squeeze(torch.stack((tokens["input_ids"],tokens["attention_mask"],tokens["token_type_ids"],),dim=2,),dim=0,)

            attr = self.attr[idx]
            return (return_tokens, attr, idx)
        
        elif self.data_name == 'MultiNLI':
            return (self.data_idx[idx], self.attr[idx], idx)
        
        else:
            if self.data_name == 'CelebA':   
                img_filename = os.path.join(self.root_dir, self.data_name, 'img_align_celeba', self.data_idx[idx])

            elif self.data_name == 'Waterbirds':
                img_filename = os.path.join(self.root_dir, self.data_name, self.data_idx[idx])
            
            img = Image.open(img_filename).convert("RGB")
            img = self.transform(img)
            attr = self.attr[idx]
            return (img, attr, idx)
        
if __name__ == "__main__":
    test_ds = My_dataset(data = 'MultiNLI', split = 'test', shuffle=True, ratio = 0.5)
    import ipdb;ipdb.set_trace()
