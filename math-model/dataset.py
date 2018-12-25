import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def divide():
    path = 'list_a.txt'
    train_path = 'train_a.txt'
    test_path = 'test_a.txt'
    with open(path) as f:
        lines = f.readlines()
    assert(len(lines) != 0)
    mapping = [x for x in lines]
    random.shuffle(mapping)
    pivot = int(len(mapping)*0.8)
    with open(train_path, 'w') as f:
        f.writelines(mapping[:pivot])
    with open(test_path, 'w') as f:
        f.writelines(mapping[pivot:])


class ModelsimDataset(Dataset):
    def __init__(self, root, file_list):
        self.root = root

        self.feature = []
        self.predict = []
        with open(file_list, 'r') as f:
            self.file_list = f.readlines()
        count = 0
        for item in self.file_list:
            print(count)
            subject = item.split(',')[0]
            other = item.split(',')[1]
            with open(root + subject, 'r') as f_subject:
                subject = f_subject.readlines()
            with open(root + other[:-1], 'r') as f_other:
                other = f_other.readlines()
            print(len(subject))
            print(len(other))
            assert(len(subject) == len(other))
            for i in range(len(subject) - 1):
                tmp = []
                for j in subject[i].split(','):
                    tmp.append(float(j))
                for j in other[i].split(','):
                    tmp.append(float(j))
                self.feature.append(tmp)
                # label
                split_result = subject[i + 1].split(',')
                self.predict.append([float(split_result[0]), float(split_result[1])])

        self.feature = np.array(self.feature)
        self.predict = np.array(self.predict)

    def __len__(self):
        return self.feature.shape[0]

    def __getitem(self, index):
        feature = self.feature[index]
        feature = torch.from_numpy(feature).float()
        predict = self.predict[index]
        predict = torch.from_numpy(predict).float()
        return feature, predict


if __name__ == "__main__":
    dataset = ModelsimDataset(root='./data/a/', file_list='test_a.txt')
    feature, predict = next(enumerate(dataset))
    print(feature.shape)
    print(predict.shape)
    print(predict)
    print(feature)
    print(len(dataset))
