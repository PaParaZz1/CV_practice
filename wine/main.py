import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from dataset import WineDataset, WineTestDataset
from models import DNN


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
            feature = feature.unsqueeze(2)
            output = model(feature)
            output = output.squeeze()

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_choice = output.data.max(dim=1)[1]
            correct = output_choice.eq(label).sum().cpu().numpy()
            print('[epoch%d: batch%d], train loss: %f, accuracy: %f' % (epoch, idx, loss.item(), correct * 1.0 / batch_size))

            if idx % 10 == 9:
                total_correct = 0.
                for index, data in enumerate(dev_dataloader):
                    feature, label = data
                    feature, label = feature.cuda(), label.cuda()
                    feature = feature.unsqueeze(2)
                    output = model(feature)
                    output = output.squeeze()
                    output_choice = output.data.max(dim=1)[1]
                    correct = output_choice.eq(label).sum().cpu().numpy()
                    total_correct += correct
                print('dev set accuracy: {}'.format(total_correct/float(dev_set_len)))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "%s/epoch_%d.pth" % (args.output_dir, epoch))


def validate(test_dataloader, model):
    result = []
    for index, data in enumerate(test_dataloader):
        feature = data
        feature = feature.cuda()
        feature = feature.unsqueeze(2)
        output = model(feature)
        output = output.squeeze()

        output_choice = output.data.max(dim=1)[1]
        output_choice = output_choice.data.cpu().numpy()
        result.append(output_choice)
    with open('submission.csv', 'w') as f:
        for item in result:
            for j in range(item.shape[0]):
                f.write(str(item[j]) + '\n') 

def main(args):
    if args.model == 'DNN':
        model = DNN(input_dim=11)
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=[50, 80], gamma=0.5)

    if args.load_path:
        if args.recover:
            load_model(model, args.load_path, strict=True)
            print('load model state dict in {}'.format(args.load_path))

    train_set = WineDataset('train_set.csv')
    dev_set = WineDataset('dev_set.csv')
    test_set = WineTestDataset('test_set.csv')
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
        validate(test_dataloader, model)
        return

    train(train_dataloader, dev_dataloader, model, optimizer, lr_scheduler, len(dev_set), args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='machine learning homework')
    parser.add_argument('--load_path', default='./experiment/conv3bn/epoch_250.pth', type=str)
    parser.add_argument('--recover', default=False, type=bool)
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--model', default='DNN', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--evaluate', default=True, type=bool)
    parser.add_argument('--loss_function', default='CrossEntropy', type=str)
    parser.add_argument('--output_dir', default='./experiment/result', type=str)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    print(args)
    main(args)
