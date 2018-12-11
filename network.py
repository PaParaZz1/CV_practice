import torch
import torch.nn as nn
from modules import ConvBlockSequential, ResidualBlock


class MouseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MouseNet, self).__init__()
        self.num_classes = num_classes
        self.activation = nn.LeakyReLU()
        self.conv1 = ConvBlockSequential(in_channels=3, out_channels=32, kernel_size=3, 
            stride=1, padding=1, init_type="kaiming", activation=self.activation, use_batchnorm=True)
        self.conv2 = ConvBlockSequential(in_channels=32, out_channels=64, kernel_size=3,
            stride=1, padding=1, init_type="kaiming", activation=self.activation, use_batchnorm=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(in_channels=64, out_channels=128, stride=2, activation=self.activation)
        self.res2 = ResidualBlock(in_channels=128, out_channels=256, stride=2, activation=self.activation)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256)
        x = self.fc(x)

        return x


class AttentionModule(nn.Module):
    def __init__(self, input_dim, use_batchnorm):
       super(AttentionModule, self).__init__()
       self.conv1 = ConvBlockSequential(in_channels=input_dim, out_channels=16, kernel_size=1, init_type="xavier", activation=nn.ReLU(), use_batchnorm=use_batchnorm)
       self.conv2 = ConvBlockSequential(in_channels=16, out_channels=input_dim, kernel_size=1, init_type="xavier", activation=nn.ReLU(), use_batchnorm=use_batchnorm)
       self.sigmoid = nn.Sigmoid()
       self.mask = nn.Sequential(self.conv1, self.conv2, self.sigmoid)

    def forward(self, x):
        mask = self.mask(x)
        x = mask * x
        return x


class MouseAttentionNet(nn.Module):
    def __init__(self, num_classes=10, use_batchnorm=True):
        super(MouseAttentionNet, self).__init__()
        self.num_classes = num_classes
        self.activation = nn.LeakyReLU()
        self.mask = AttentionModule(input_dim=3, use_batchnorm=use_batchnorm)
        self.conv1 = ConvBlockSequential(in_channels=3, out_channels=32, kernel_size=3, 
            stride=1, padding=1, init_type="xavier", activation=self.activation, use_batchnorm=use_batchnorm)
        self.conv2 = ConvBlockSequential(in_channels=32, out_channels=64, kernel_size=3,
            stride=1, padding=1, init_type="kaiming", activation=self.activation, use_batchnorm=use_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(in_channels=64, out_channels=128, stride=2, activation=self.activation, use_batchnorm=use_batchnorm)
        self.res2 = ResidualBlock(in_channels=128, out_channels=256, stride=2, activation=self.activation, use_batchnorm=use_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=4)
        self.fc = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = self.mask(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), 256)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    net = MouseAttentionNet().cuda()
    inputs = torch.randn(4, 3, 32, 32).cuda()
    out = net(inputs)
    print(out.shape)
    print(out)
        

