import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def divide():
    path = '/mnt/lustre/niuyazhe/nyz/ml/team/data/train_img.csv'
    label_path = '/mnt/lustre/niuyazhe/nyz/ml/team/data/train_labels.csv'
    train_set = '/mnt/lustre/niuyazhe/nyz/ml/team/data/train_set.csv'
    dev_set = '/mnt/lustre/niuyazhe/nyz/ml/team/data/dev_set.csv'
    with open(path) as f:
        lines = f.readlines()
    with open(label_path) as f:
        label_lines = f.readlines()
    assert(len(lines) != 0)
    assert(len(lines) == len(label_lines))
    mapping = [lines, label_lines]
    mapping = [(lines[x][:-1] + ',' + label_lines[x][:-2] + '\n') for x in range(len(lines))]
    
    random.shuffle(mapping)
    pivot = int(len(mapping)*0.8)
    with open(train_set, 'w') as f:
        f.writelines(mapping[:pivot])
    with open(dev_set, 'w') as f:
        f.writelines(mapping[pivot:])

    
class DigitDataset(Dataset):
    def __init__(self, img_file_name, root_path='/mnt/lustre/niuyazhe/nyz/ml/team/data/train/', transform=None):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()])
        else:
            self.transform = transform
        self.imgs = []
        self.labels = []
        with open(os.path.join(root_path, img_file_name)) as f:
            lines = f.readlines()
            for line in lines:
                result_split = line.split(',')
                self.imgs.append(root_path + result_split[0])
                self.labels.append(int(float(result_split[1])))
        assert(len(self.imgs) != 0)

        self.labels = np.array(self.labels)

    def loader(self, path):
        return Image.open(path).convert('RGB')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.loader(self.imgs[index])
        img = self.transform(img)
        label = self.labels[index]
        label = np.array(label)
        label = torch.from_numpy(label).long()
        return img, label


class DigitTestDataset(Dataset):
    def __init__(self, img_file_name, root_path='/mnt/lustre/niuyazhe/nyz/ml/team/data/test/', transform=None):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()])
        else:
            print('use self-defined data transform')
            self.transform = transform
        self.imgs = []
        with open(os.path.join(root_path, img_file_name)) as f:
            lines = f.readlines()
            for line in lines:
                self.imgs.append(root_path + line[:-1])
        assert(len(self.imgs) != 0)

    def loader(self, path):
        return Image.open(path).convert('RGB')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.loader(self.imgs[index])
        img = self.transform(img)
        return img


if __name__ == "__main__":
    test_dataset = DigitTestDataset('../test_set.csv')
    img = test_dataset[0]
    print(img.shape)
    print(img)
    # divide()
