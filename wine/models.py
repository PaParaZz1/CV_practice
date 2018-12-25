import torch
import torch.nn as nn

from torch.nn.init import xavier_normal_, kaiming_normal_

def WeightInitialize(init_type, weight, activation = None):
	if init_type is None:
		return
	def XavierInit(weight, activation):
		xavier_normal_(weight)
	def KaimingInit(weight, activation):
		assert not activation is None
		if hasattr(activation, "negative_slope"):
			kaiming_normal(weight, a = activation.negative_slope)
		else:
			kaiming_normal(weight, a = 0)

	init_type_dict = {"xavier" : XavierInit, "kaiming" : KaimingInit}
	if init_type in init_type_dict:
		init_type_dict[init_type](weight, activation)
	else:
		raise KeyError("Invalid Key:%s" % init_type)

def ConvBlock(in_channels, out_channels, kernel_size, stride = 1, padding = 0, init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = False):
	block_conv = []
	# (TODO) deal with special(2D) padding
	block_conv.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding))
	WeightInitialize(init_type, block_conv[-1].weight, activation)
	if not activation is None:
		block_conv.append(activation)
	if use_batchnorm:
		block_conv.append(nn.BatchNorm1d(out_channels))
	return block_conv

def ConvBlockSequential(in_channels, out_channels, kernel_size, stride = 1, padding = 0, init_type = "kaiming", activation = nn.ReLU(), use_batchnorm = False):
	seq = nn.Sequential(*ConvBlock(in_channels, out_channels, kernel_size, stride, padding, init_type, activation, use_batchnorm))
	seq.out_channels = out_channels
	return seq

class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.activation = nn.LeakyReLU()
        '''
        fc3
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, 2)
        
        self.conv1 = ConvBlockSequential(in_channels=input_dim, out_channels=100, kernel_size=1, init_type="xavier", activation=self.activation, use_batchnorm=True)
        self.conv2 = ConvBlockSequential(in_channels=100, out_channels=64, kernel_size=1, init_type="xavier", activation=self.activation, use_batchnorm=True)
        self.conv3 = ConvBlockSequential(in_channels=64, out_channels=2, kernel_size=1, init_type="xavier", activation=None, use_batchnorm=True)
        '''
        self.conv1 = ConvBlockSequential(in_channels=input_dim, out_channels=64, kernel_size=1, init_type="xavier", activation=self.activation, use_batchnorm=True)
        self.conv2 = ConvBlockSequential(in_channels=64, out_channels=100, kernel_size=1, init_type="xavier", activation=self.activation, use_batchnorm=True)
        self.conv3 = ConvBlockSequential(in_channels=100, out_channels=64, kernel_size=1, init_type="xavier", activation=self.activation, use_batchnorm=True)
        self.conv4 = ConvBlockSequential(in_channels=64, out_channels=32, kernel_size=1, init_type="xavier", activation=self.activation, use_batchnorm=True)
        self.conv5 = ConvBlockSequential(in_channels=32, out_channels=2, kernel_size=1, init_type="xavier", activation=None, use_batchnorm=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
