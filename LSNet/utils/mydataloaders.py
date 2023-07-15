import os
import torch.utils.data as data
from PIL import Image
from utils import transforms as tr
import cv2
import numpy as np
'''
Load all training and validation data paths
'''


def full_path_loader(data_dir):
    train_data = [i for i in os.listdir(data_dir + '/train/line/B/') if not
    i.startswith('.')]
    train_data.sort()

    valid_data = [i for i in os.listdir(data_dir + '/val/line/B/') if not
    i.startswith('.')]
    valid_data.sort()

    train_label_paths = []
    val_label_paths = []
    for img in train_data:
        train_label_paths.append(data_dir + '/train/line/OUT/' + img)
    for img in valid_data:
        val_label_paths.append(data_dir + '/val/line/OUT/' + img)

    train_data_path = []
    val_data_path = []

    for img in train_data:
        train_data_path.append([data_dir + '/train/line/', img])
    for img in valid_data:
        val_data_path.append([data_dir + '/val/line/', img])

    train_dataset = {}
    val_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp],
                             'label': train_label_paths[cp]}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],
                           'label': val_label_paths[cp]}

    return train_dataset, val_dataset


'''
Load all testing data paths
'''


def full_test_loader(data_dir):
    test_data = [i for i in os.listdir(data_dir + '/test/line/B/') if not
    i.startswith('.')]
    test_data.sort()

    test_label_paths = []
    for img in test_data:
        test_label_paths.append(data_dir + '/test/line/OUT/' + img)

    test_data_path = []
    for img in test_data:
        test_data_path.append([data_dir + '/test/line/', img])

    test_dataset = {}
    for cp in range(len(test_data)):
        test_dataset[cp] = {'image': test_data_path[cp],
                            'label': test_label_paths[cp]}

    return test_dataset


def cdd_loader(img_path, label_path, aug):
    dir = img_path[0]
    name = img_path[1]
    ref_id = int(int(name.split('.')[0]) / 500)
    ref_id = ref_id if int(name.split('.')[0]) % 500 == 0 else ref_id + 1
    img1 = Image.open(dir + 'ref/' + str(ref_id)+'.png')
    img2 = Image.open(dir + 'B/' + name)
    label = Image.open(label_path)
    ###
    label = np.array(label)
    label[label!=0]=255
    label = Image.fromarray(label)
    ###
    sample = {'image': (img1, img2), 'label': label}

    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    return sample['image'][0], sample['image'][1], sample['label'], name


class CDDloader(data.Dataset):

    def __init__(self, full_load, aug=False):
        self.full_load = full_load
        self.loader = cdd_loader
        self.aug = aug

    def __getitem__(self, index):
        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']

        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)