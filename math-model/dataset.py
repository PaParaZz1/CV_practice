import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def divide():
    path = '/mnt/lustre/niuyazhe/nyz/ml/team/data/test_set.csv'
    label_path = '/mnt/lustre/niuyazhe/nyz/ml/team/data/test_labels.csv'
    train_set = '/mnt/lustre/niuyazhe/nyz/ml/team/data/train_set.csv'
    dev_set = '/mnt/lustre/niuyazhe/nyz/ml/team/data/dev_set.csv'
    test_set = '/mnt/lustre/niuyazhe/nyz/ml/team/data/test_dev_set.csv'
    with open(path) as f:
        lines = f.readlines()
    with open(label_path) as f:
        label_lines = f.readlines()
    assert(len(lines) != 0)
    assert(len(lines) == len(label_lines))
    mapping = [lines, label_lines]
    mapping = [(lines[x][:-1] + ',' + label_lines[x][:-1] + '\n') for x in range(len(lines))]
    '''
    random.shuffle(mapping)
    pivot = int(len(mapping)*0.8)
    with open(train_set, 'w') as f:
        f.writelines(mapping[:pivot])
    with open(dev_set, 'w') as f:
        f.writelines(mapping[pivot:])
    '''
    with open(test_set, 'w') as f:
        f.writelines(mapping)

class ModelsimDataset(Dataset):
    def __init__(self):

    def __len(self):

    def __getitem(self, index):
