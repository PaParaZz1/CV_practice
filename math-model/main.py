import os
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from dataset import ModelsimDataset, ModelsimTestDataset
from network import SFNet


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


def train(train_dataloader, dev_dataloader, model, optimizer, lr_scheduler, args):
    model.train()
    if args.loss_function == 'L1':
        criterion = nn.L1Loss()
    elif args.loss_function == 'L2':
        criterion = nn.MSELoss()
    else:
        raise ValueError("invalid lossfunction: {}".format(args.loss_function))
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

            print('[epoch%d: batch%d], train loss: %f' % (epoch, idx, loss.item()))

            if idx % 50 == 49:
                total_error = 0.
                for index, data in enumerate(dev_dataloader):
                    feature, label = data
                    feature, label = feature.cuda(), label.cuda()
                    output = model(feature)
                    test_error = torch.norm(label - output, p=2)
                    total_error += test_error
                print('test set error: {}'.format(total_error))
        if epoch % 2 == 0:
            torch.save(model.state_dict(), "%s/epoch_%d.pth" %
                       (args.output_dir, epoch))


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
    if args.model == 'SFNet':
        model = SFNet(input_dim=12)
    else:
        raise ValueError("input network type: {}".format(args.model))
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 # momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=[100, 200], gamma=0.5)

    if args.load_path:
        if args.recover:
            load_model(model, args.load_path, strict=True)
            print('load model state dict in {}'.format(args.load_path))

    train_set = ModelsimDataset(root=args.root, file_list='train_a.txt')
    test_set = ModelsimTestDataset(root=args.root, file_list='test_a.txt')
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

    if args.evaluate:
        validate(test_dataloader, model)
        return

    train(train_dataloader, test_dataloader, model,
          optimizer, lr_scheduler, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='math model homework')
    parser.add_argument(
        '--load_path', default='./experiment/SFNet/epoch_63.pth', type=str)
    parser.add_argument('--root', default='./data/a/')
    parser.add_argument('--recover', default=False, type=bool)
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--model', default='SFNet', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--loss_function', default='L2', type=str)
    parser.add_argument('--output_dir', default='experiment/result', type=str)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    print(args)
    main(args)
