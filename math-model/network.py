import torch
import torch.nn as nn
from modules import ConvBlockSequential


class SFNet(nn.Module):
    def __init__(self, input_dim, node_list=None):
        super(SFNet, self).__init__()
        if node_list is None:
            self.node_list = [128, 64, 32]
        else:
            self.node_list = node_list

        self.layer_number = len(self.node_list)
        self.activation = nn.Sigmoid()
        self.use_bn = False

        network = []
        layer_input_dim = input_dim
        for i in range(self.layer_number):
            network.append(ConvBlockSequential(in_channels=layer_input_dim,
                                               out_channels=self.node_list[i],
                                               kernel_size=1,
                                               init_type="xavier",
                                               activation=self.activation,
                                               use_batchnorm=self.use_bn))
            layer_input_dim = self.node_list[i]
        self.main = nn.Sequential(*network)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(self.node_list[-1], 2)

    def forward(self, x):
        x = self.main(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    net = SFNet(12).cuda()
    inputs = torch.randn(8, 12, 2).cuda()
    out = net(inputs)
    print(out.shape)
    print(out)
