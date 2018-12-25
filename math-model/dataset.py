import os
import random
import numpy as np
#mport torch
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
    def __init__(self):
        self.a = a

    def __len(self):
        return 0

    def __getitem(self, index):
        return 0


if __name__ == "__main__":
    divide()
