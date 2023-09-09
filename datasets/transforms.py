"""
Author: Austin Dibble
"""

from torchvision.transforms import Normalize
from torch.utils.data import Dataset
import torch

class NormalizeScale:
    def __init__(self, scale_factor=10000):
        super().__init__()
        self.scale_factor = scale_factor

    def __call__(self, sample):
        sample['image'] = sample['image'].to(torch.float) / self.scale_factor
        return sample

class NormalizeImageDict(Normalize):
    def __init__(self, mean, std):
        super().__init__(mean, std)

    def __call__(self, sample):
        sample['image'] = super().__call__(sample['image'].to(torch.float))
        return sample

class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x = self.subset[index]
        if self.transform:
            return self.transform(x)
        else:
            return x

    def __len__(self):
        return len(self.subset)
