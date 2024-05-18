import torch
import torch.nn as nn
from nfn import layers
from nfn.common import network_spec_from_wsfeat

class HyperNetworkMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HyperNetworkMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim, bias=False)
        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HyperNetworkConv(nn.Module):
    def __init__(self, input_dim, num_kernels=256, kernel_width=2, stride=2):
        super(HyperNetwork, self).__init__()
        self.num_kernels = num_kernels
        self.conv1 = nn.Conv1d(2, 16, 3)
        self.pool1 = nn.MaxPool1d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, 3)
        self.pool2 = nn.MaxPool1d(2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(32, 1, 1)
        self.up = nn.Upsample(input_dim//2)

    def forward(self, x):
        x = torch.stack([x])
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.up(x)
        x = self.conv3(x)

        return x.squeeze()

class HypernetNFN(nn.Module):
    def __init__(self, network_spec, nfn_channels = 32, input_channels=2):
        super(HypernetNFN, self).__init__()
        self.network_spec = network_spec
        self.nfn_channels = nfn_channels
        self.input_channels = input_channels

        self.NP1 = layers.NPLinear(network_spec, self.input_channels, nfn_channels, io_embed=True)
        self.relu1 = layers.TupleOp(nn.ReLU())
        self.NP2 = layers.NPLinear(network_spec, nfn_channels, nfn_channels, io_embed=True)
        self.relu2 = layers.TupleOp(nn.ReLU())
        self.NP3 = layers.NPLinear(network_spec, nfn_channels, 1, io_embed=True)

        self.pool = layers.HNPPool(network_spec)
        self.flatten = nn.Flatten(start_dim=-2)
        self.fc = nn.Linear(nfn_channels * layers.HNPPool.get_num_outs(network_spec), 1)

    def forward(self, x):
        x = self.NP1(x)
        x = self.relu1(x)
        x = self.NP2(x)
        x = self.relu2(x)
        x = self.NP3(x)
        #x = self.pool(x)
        #x = self.flatten(x)
        #x = self.fc(x)

        return x