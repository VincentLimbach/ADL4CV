import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=16):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        frequency = torch.exp(torch.linspace(0, -math.log(10000.0), d_model // 2))
        self.register_buffer('frequency', frequency)

    def forward(self, x):
        batch_size, _ = x.shape
        
        x_enc = x[:, :1] * self.frequency * math.pi
        y_enc = x[:, 1:] * self.frequency * math.pi
        enc_x = torch.cat((torch.sin(x_enc), torch.cos(x_enc)), dim=-1)
        enc_y = torch.cat((torch.sin(y_enc), torch.cos(y_enc)), dim=-1)
        
        encoded_input = torch.cat((enc_x, enc_y), dim=-1)
        return encoded_input

class MLP(nn.Module):
    def __init__(self, positional=False):
        super(MLP, self).__init__()
        if positional:
            self.positional_encoding = PositionalEncoding(d_model=16)
        else:
            self.linear1 = nn.Linear(2, 32)
            self.relu1 = nn.ReLU()
        
        self.linear2 = nn.Linear(32, 64)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(64, 64)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(64, 1)
        
        self.positional = positional

    def forward(self, x):
        if self.positional:
            x = self.positional_encoding(x)
        else:
            x = self.linear1(x)
            x = self.relu1(x)
        
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

class sMLP(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, layers=2, positional=True):
        super(sMLP, self).__init__()
        self.positional = positional
        if self.positional:
            self.positional_encoding = PositionalEncoding()

        self.layers = nn.ModuleList()
        self.layer_sizes = []
        current_features = input_features

        # Create layers and record parameter sizes for each layer
        for i in range(layers):
            next_features = hidden_features if i < layers - 1 else output_features
            self.layers.append(nn.Linear(current_features, next_features))
            self.layer_sizes.append((current_features * next_features + next_features))  # Weight + Bias
            current_features = next_features

        self.activations = nn.ModuleList([nn.LeakyReLU() for _ in range(layers)])

    def forward(self, x, external_parameters=None):
        if self.positional:
            x = self.positional_encoding(x)

        param_index = 0  # Start index for slicing parameters
        for i, (linear, activation) in enumerate(zip(self.layers, self.activations)):
            if external_parameters is not None:
                # Extract the weights and biases for this layer
                weight_size = linear.weight.size(0) * linear.weight.size(1)
                bias_size = linear.bias.size(0)

                weight = external_parameters[param_index:param_index + weight_size].view_as(linear.weight)
                bias = external_parameters[param_index + weight_size:param_index + weight_size + bias_size].view_as(linear.bias)
                param_index += weight_size + bias_size

                x = F.linear(x, weight, bias)
            else:
                x = linear(x)
            x = activation(x)

        return x

class HyperNetwork(nn.Module):
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

#input_dim = 10
#x = torch.randn(input_dim) 
#model = HyperNetwork(input_dim)
#output = model(x)
#print(output)

