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
        self.near = 5

        self.feature = []
        self.predict = []
        with open(file_list, 'r') as f:
            self.file_list = f.readlines()
        for item in self.file_list:
            subject = item.split(',')[0]
            other = item.split(',')[1]
            with open(root + subject, 'r') as f_subject:
                subject = f_subject.readlines()
            with open(root + other[:-1], 'r') as f_other:
                other = f_other.readlines()
            assert(len(subject) == len(other))
            for i in range(len(subject) - 1):
                tmp = []
                subject_split = subject[i].split(',')
                other_split = other[i].split(',')
                for j in range(len(subject_split)):
                    tmp.append(float(subject_split[j]))
                for j in range(self.near):
                    value = float(other_split[j])
                    value = np.clip(value, -3, 3)
                    value /= 3
                    tmp.append(value)
                    value = float(other_split[j + self.near])
                    value = np.clip(value, -3, 3)
                    value /= 3
                    tmp.append(value)
                for j in range(2*self.near, 3*self.near):
                    tmp.append(float(other_split[j]))
                    tmp.append(float(other_split[j + self.near]))
                self.feature.append(tmp)
                # label
                split_result = subject[i + 1].split(',')
                self.predict.append([float(split_result[0]), float(split_result[1])])

        self.feature = np.array(self.feature)
        self.feature = self.feature.reshape(-1, 12, 2)
        self.predict = np.array(self.predict)

    def __len__(self):
        return self.feature.shape[0]

    def __getitem__(self, index):
        feature = self.feature[index]
        feature = torch.from_numpy(feature).float()
        predict = self.predict[index]
        predict = torch.from_numpy(predict).float()
        return feature, predict


class ModelsimTestDataset(Dataset):
    def __init__(self, root, file_list):
        self.root = root

    def __len__(self):
        return 0

    def __getitem(self):
        return 0


if __name__ == "__main__":
    dataset = ModelsimDataset(root='./data/a/', file_list='test_a.txt')
    print(dataset.feature.shape)
    print(dataset.feature[0].shape)
