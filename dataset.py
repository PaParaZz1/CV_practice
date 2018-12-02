import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def divide():
    path = '/mnt/lustre/niuyazhe/nyz/ml/person/data/train.csv'
    train_set = '/mnt/lustre/niuyazhe/nyz/ml/person/data/train_set.csv'
    dev_set = '/mnt/lustre/niuyazhe/nyz/ml/person/data/dev_set.csv'
    with open(path) as f:
        lines = f.readlines()
    assert(len(lines) != 0)
    
    random.shuffle(lines)
    pivot = int(len(lines)*0.8)
    with open(train_set, 'w') as f:
        f.writelines(lines[:pivot])
    with open(dev_set, 'w') as f:
        f.writelines(lines[pivot:])

    
class WineDataset(Dataset):
    def __init__(self, file_name, root_path='/mnt/lustre/niuyazhe/nyz/ml/person/data/'):
        self.feature = []
        self.label = []
        with open(os.path.join(root_path, file_name)) as f:
            lines = f.readlines()
            for line in lines:
                line = line[:-1]
                line = line.split(',')
                label = np.array((int)(line[-1]))
                self.label.append(label)
                feature = line[:-1]
                feature = np.array(feature).astype(np.float32)
                self.feature.append(feature)
        assert(len(self.feature) != 0)
        assert(len(self.feature) == len(self.label))

        self.feature = np.array(self.feature)
        print(self.feature.shape)
        max_val = self.feature.max(axis=0)
        min_val = self.feature.min(axis=0)
        self.feature = (self.feature - min_val) / (max_val - min_val)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        feature = self.feature[index]
        label = self.label[index]
        feature = torch.from_numpy(feature).float()
        label = torch.from_numpy(label).long()
        return feature, label

    def distribution(self):
        total = len(self.label)
        label_np = np.array(self.label)
        positive = label_np.sum()
        print('positive: {}, negative: {}'.format(positive*1.0 / total, 1-(positive*1.0 / total)))


class WineTestDataset(Dataset):
    def __init__(self, file_name, root_path='/mnt/lustre/niuyazhe/nyz/ml/person/data/'):
        self.feature = []
        with open(os.path.join(root_path, file_name)) as f:
            lines = f.readlines()
            for line in lines:
                line = line[:-1]
                line = line.split(',')
                feature = line
                feature = np.array(feature).astype(np.float32)
                self.feature.append(feature)
        assert(len(self.feature) != 0)

        self.feature = np.array(self.feature)
        print(self.feature.shape)
        max_val = self.feature.max(axis=0)
        min_val = self.feature.min(axis=0)
        self.feature = (self.feature - min_val) / (max_val - min_val)

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):
        feature = self.feature[index]
        feature = torch.from_numpy(feature).float()
        return feature
if __name__ == "__main__":
    test_dataset = WineDataset('train.csv')
    #print(test_dataset[0])
    #divide()
    test_dataset.distribution()
