"""
Author: Austin Dibble
"""

import os
import shutil
import zipfile
import torch
import matplotlib.pyplot as plt
from typing import Dict, Optional
from torchvision.transforms.functional import to_pil_image
from torchgeo.datasets.utils import draw_semantic_segmentation_masks
import numpy as np
import kornia.augmentation as K

import multiprocessing as mp

class SharedObjectCache:
    """
    A simple shared memory cache for arbitrary Python objects.
    This cache is backed by a multiprocessing Manager's dict, allowing
    it to be shared between multiple processes.
    """

    def __init__(self):
        # Create a manager and a dict proxy for sharing the cache across processes
        self.manager = mp.Manager()
        self.cache = self.manager.dict()

    def set(self, key, value):
        """Set a value in the cache for a given key."""
        self.cache[key] = value

    def get(self, key):
        """Retrieve a value from the cache for a given key.
        
        Returns None if the key is not present.
        """
        return self.cache.get(key, None)

    def remove(self, key):
        """Remove a key from the cache if it exists."""
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        """Clear the entire cache."""
        self.cache.clear()

    def keys(self):
        """Return a list of all keys in the cache."""
        return list(self.cache.keys())

    def __len__(self):
        """Return the number of items in the cache."""
        return len(self.cache)

    def __getitem__(self, key):
        """Enable access using square brackets (e.g., cache[key])."""
        return self.get(key)

    def __setitem__(self, key, value):
        """Enable setting values using square brackets (e.g., cache[key] = value)."""
        self.set(key, value)

    def __delitem__(self, key):
        """Enable deletion of items using square brackets (e.g., del cache[key])."""
        self.remove(key)

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def copy_file(source_path, destination_path):
    shutil.copy(source_path, destination_path)

def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def move_folder(source_path, destination_path):
    shutil.move(source_path, destination_path)

def copy_and_unzip(local_path, drive_path, archive_name):
    print(f"Copying the archive file '{archive_name}' from Google Drive to local folder...")
    copy_file(os.path.join(drive_path, archive_name), os.path.join(local_path, archive_name))
    print("Archive file copied successfully.")

    print(f"Unzipping the archive file '{archive_name}'...")
    unzip_file(os.path.join(local_path, archive_name), local_path)
    print("Archive file unzipped successfully.")

def load_and_prepare_oscd(local_path="/content/datasets", drive_path="/content/drive/MyDrive/2023_dissertation/dataset_archives/"):
    if os.path.exists(os.path.join(local_path, "OSCD")):
        print("An 'OSCD' folder already exists in the local path. Skipping dataset loading and preparation.")
        return
    
    create_folder_if_not_exists(local_path)

    OSCD = "OSCD_Daudt_2018_full.zip"
    copy_and_unzip(local_path, drive_path, OSCD)

    print("Moving the extracted folders and renaming if necessary...")
    move_folder(os.path.join(local_path, "Onera"), os.path.join(local_path, "OSCD"))
    print("Main folder renamed successfully.")

    # Define the folder mappings
    folder_mappings = {
        "images": "Onera Satellite Change Detection dataset - Images",
        "train_labels": "Onera Satellite Change Detection dataset - Train Labels",
        "test_labels": "Onera Satellite Change Detection dataset - Test Labels"
    }

    # Move and rename the extracted folders
    oscd_path = os.path.join(local_path, "OSCD")
    for source_folder, destination_folder in folder_mappings.items():
        source_path = os.path.join(oscd_path, source_folder)
        destination_path = os.path.join(oscd_path, destination_folder)
        print(f"Renaming folder '{source_folder}' to '{destination_folder}'...")
        move_folder(source_path, destination_path)
        print(f"Folder '{source_folder}' renamed successfully to '{destination_folder}'.")
    
    print("Dataset loading and preparation complete.")

def load_and_prepare_omcd(local_path="/content/datasets", drive_path="/content/drive/MyDrive/2023_dissertation/dataset_archives/"):
    if os.path.exists(os.path.join(local_path, "OMCD")):
        print("An 'OMCD' folder already exists in the local path. Skipping dataset loading and preparation.")
        return

    create_folder_if_not_exists(local_path)

    OMCD = "OMCD_Li_2023.zip"
    copy_and_unzip(local_path, drive_path, OMCD)

    print("Moving the extracted folders and renaming to OMCD...")
    move_folder(os.path.join(local_path, "open-pit mine change detection dataset"), os.path.join(local_path, "OMCD"))
    print("Main folder renamed successfully.")

    print("Dataset loading and preparation complete.")

def crop_sample(dataset, index: int, size: int = 256):
    from torchvision.transforms import CenterCrop

    sample = dataset[index]
    cropper = CenterCrop(size)

    sample['image'] = cropper(sample['image'])
    sample['mask'] = cropper(sample['mask'])

    return sample

def normalize_sample(sample, mean, std):
    image = sample['image'].float()
    if len(image.shape) < 4:
        image = image.unsqueeze(0)
    normalize = K.Normalize(mean, std)
    normalized_image = normalize(image)
    sample['image'] = normalized_image
    return sample

def get_oscd_norm_coefficients(bands="rgb"):
    # mean = OSCDDataModule.mean
    # std = OSCDDataModule.std
    mean = torch.tensor([1571.1372, 1365.5087, 1284.8223, 1298.9539, 1431.2260, 1860.9531,
                    2081.9634, 1994.7665, 2214.5986,  641.4485,   14.3672, 1957.3165,
                    1419.6107])
    std =  torch.tensor([274.9591,  414.6901,  537.6539,  765.5303,  724.2261,  760.2133,
                    848.7888,  866.8081,  920.1696,  322.1572,    8.6878, 1019.1249,
                    872.1970])
    if bands == "rgb":
        mean = mean[[3, 2, 1]]
        std = std[[3, 2, 1]]

    mean = torch.cat([mean, mean], dim=0)
    std = torch.cat([std, std], dim=0)
    return mean, std

def get_omcd_norm_coefficients():
    # mean = OSCDDataModule.mean
    # std = OSCDDataModule.std
    mean = torch.tensor([121.7963, 123.6833, 116.5527])
    std = torch.tensor([67.2949, 68.0268, 65.1758])

    mean = torch.cat([mean, mean], dim=0)
    std = torch.cat([std, std], dim=0)
    return mean, std