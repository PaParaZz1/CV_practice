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

def find_peak(th_sum, min_limit, max_limit):
    threshold = 1.2
    th_slice = th_sum[min_limit:max_limit+1]
    th_mean = th_slice.mean()
    th_max = th_slice.max()
    th_min = th_slice.min()
    if th_max > th_mean * threshold:
        return th_slice.tolist().index(th_max) + min_limit
    if th_min < th_mean / threshold:
        return th_slice.tolist().index(th_min) + min_limit
    return -1


def segment(filename, norm=False, store=False, store_root='./target_fine/', resize=False, no_segment=False):
    if store and not os.path.exists(store_root):
        os.mkdir(store_root)
    img = Image.open(filename).convert('RGB')
    img_np = np.array(img)
    if no_segment:
        img_float_np = np.float32(img_np)
        return img_float_np
    gray = cv2.cvtColor(img_np,cv2.COLOR_RGB2GRAY)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th_sum = th.sum(axis=0)
    min_left = 8
    max_left = 12
    min_right = 20
    max_right = 24
    
    left = find_peak(th_sum, min_left, max_left)
    right = find_peak(th_sum, min_right, max_right)
    if left == -1:
        left = min_left
    if right == -1:
        right = max_right
    img_np[:,:left-1] = 0
    img_np[:,right+1:] = 0
    #img_save_np = cv2.resize(img_np[:,left-1:right+1], (32,32))
    img_save_np = img_np
    img_float_np = np.float32(img_np)
    if norm:
        img_slice = img_float_np[:,left-1:right+1]
        img_mean = img_slice.mean(axis=1)
        img_mean = img_mean.mean(axis=0)
        img_std = img_slice.std(axis=1)
        img_std = img_std.std(axis=0)
        img_float_np[:,left-1:right+1] = (img_slice-img_mean) / img_std
    if store:
        img = Image.fromarray(np.uint8(img_save_np))
        img.save(os.path.join(store_root + filename.split('.')[0] + '_seg.jpg'))
    if resize:
        img_float_np = cv2.resize(img_float_np[:,left-1:right+1], (32,32))
    return img_float_np

    
class DigitDataset(Dataset):
    def __init__(self, img_file_name, root_path='/mnt/lustre/niuyazhe/nyz/ml/team/data/argument_train/pre/', transform=None):
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()])
        else:
            self.transform = transform
        self.imgs = []
        self.labels = []
        if isinstance(img_file_name, list):
            for item in img_file_name:
                with open(os.path.join(root_path, item)) as f:
                    lines = f.readlines()
                    for line in lines:
                        result_split = line.split(',')
                        self.imgs.append(root_path + result_split[0])
                        self.labels.append(int(float(result_split[1])))
        else:
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

    def loader_segment(self, path):
        return segment(path, no_segment=False, resize=True)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.loader_segment(self.imgs[index])
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
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
        #img = self.transform(img)
        return img


if __name__ == "__main__":
    test_dataset = DigitDataset('../data.csv')
    img = test_dataset[0]
    print(img[0].shape)
    print(img[0])
    print(img[0].std())
    print(img[0].max())
    #divide()
