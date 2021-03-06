import os
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

from dataset_new import DigitDataset, DigitTestDataset
from resnet_interface import Iresnet
from network import MouseNet, MouseAttentionNet


def load_model(model, path, strict=False):
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key[:6] == 'module':
            name = key[7:]  # remove 'module'
        else:
            name = key
        new_state_dict[name] = val
    model.load_state_dict(new_state_dict, strict=strict)
    all_keys = set(new_state_dict.keys())
    actual_keys = set(model.state_dict().keys())
    missing_keys = actual_keys - all_keys
    for k in missing_keys:
        print(k)


def train(train_dataloader, dev_dataloader, model, optimizer, lr_scheduler, dev_set_len, args):
    model.train()
    if args.loss_function == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    batch_size = args.batch_size
    print('batch_size: {}'.format(batch_size))

    for epoch in range(args.epoch):
        lr_scheduler.step()
        current_lr = lr_scheduler.get_lr()[0]
        print('current_lr: {}'.format(current_lr))
        for idx, data in enumerate(train_dataloader):
            feature, label = data
            feature, label = feature.cuda(), label.cuda()
            cur_length = label.shape[0]
            output = model(feature)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_choice = output.data.max(dim=1)[1]
            correct = output_choice.eq(label).sum().cpu().numpy()
            print('[epoch%d: batch%d], train loss: %f, accuracy: %f' % (epoch, idx, loss.item(), correct * 1.0 / cur_length))

            if idx % 50 == 49:
                total_correct = 0.
                for index, data in enumerate(dev_dataloader):
                    feature, label = data
                    feature, label = feature.cuda(), label.cuda()
                    output = model(feature)
                    output_choice = output.data.max(dim=1)[1]
                    correct = output_choice.eq(label).sum().cpu().numpy()
                    total_correct += correct
                print('dev set accuracy: {}'.format(total_correct/float(dev_set_len)))
        if epoch % 1 == 0:
            torch.save(model.state_dict(), "%s/epoch_%d.pth" % (args.output_dir, epoch))


def validate(test_dataloader, model):
    result = []
    for index, data in enumerate(test_dataloader):
        feature = data
        feature = feature.cuda()
        output = model(feature)

        output_choice = output.data.max(dim=1)[1]
        output_choice = output_choice.data.cpu().numpy()
        result.append(output_choice)
    with open('submission.csv', 'w') as f:
        for item in result:
            for j in range(item.shape[0]):
                f.write(str(item[j]) + '\n') 

def main(args):
    if args.model == 'resnet18':
        model = Iresnet(pretrained=True, backbone_type='resnet18')
    elif args.model == 'MouseNet':
        model = MouseNet()
    elif args.model == 'MouseAttentionNet':
        model = MouseAttentionNet(use_batchnorm=False)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                #momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=[100, 200], gamma=0.5)

    if args.load_path:
        if args.recover:
            load_model(model, args.load_path, strict=True)
            print('load model state dict in {}'.format(args.load_path))

    resize_size = args.resize_size
    transform = transforms.Compose([
                    transforms.Resize(resize_size, resize_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5,), std=(0.5, 0.5, 0.5))])
    train_set = DigitDataset('../data.csv', transform=transform)
    dev_set = DigitDataset('../dev_set.csv', root_path='/mnt/lustre/niuyazhe/nyz/ml/team/data/train/', transform=transform)
    test_set = DigitTestDataset('../test_set.csv', transform=transform)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
    dev_dataloader = DataLoader(dev_set, batch_size=args.batch_size, 
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, 
                                 shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

    if args.evaluate:
        validate(dev_dataloader, model)
        return

    train(train_dataloader, dev_dataloader, model, optimizer, lr_scheduler, len(dev_set), args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='machine learning homework')
    parser.add_argument('--load_path', default='./experiment/premousenet_scratch_resize/epoch_63.pth', type=str)
    parser.add_argument('--recover', default=True, type=bool)
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--evaluate', default=True, type=bool)
    parser.add_argument('--loss_function', default='CrossEntropy', type=str)
    parser.add_argument('--output_dir', default='./experiment/result', type=str)
    parser.add_argument('--resize_size', default=32, type=int)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    print(args)
    main(args)
