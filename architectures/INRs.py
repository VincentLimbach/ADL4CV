import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math 

from abc import ABC, abstractmethod

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class LearnableFourierEncoding(nn.Module):
    def __init__(self, d_model=16):
        super(LearnableFourierEncoding, self).__init__()
        self.d_model = d_model
        self.freqs = nn.Parameter(torch.randn(d_model // 2))
        self.phases = nn.Parameter(torch.zeros(d_model // 2))

    def forward(self, x):
        batch_size, _ = x.shape
        x_enc = x[:, :1] * self.freqs * math.pi + self.phases
        y_enc = x[:, 1:] * self.freqs * math.pi + self.phases
        enc_x = torch.cat((torch.sin(x_enc), torch.cos(x_enc)), dim=-1)
        enc_y = torch.cat((torch.sin(y_enc), torch.cos(y_enc)), dim=-1)
        encoded_input = torch.cat((enc_x, enc_y), dim=-1)
        return encoded_input

class PositionalEncodingOld(nn.Module):
    def __init__(self, d_model=16):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.frequency = torch.exp(torch.linspace(0, -math.log(10000.0), d_model // 2))
        #self.register_buffer('frequency', frequency)

    def forward(self, x):
        batch_size, _ = x.shape

        x_enc = x[:, :1] * self.frequency * math.pi
        y_enc = x[:, 1:] * self.frequency * math.pi
        enc_x = torch.cat((torch.sin(x_enc), torch.cos(x_enc)), dim=-1)
        enc_y = torch.cat((torch.sin(y_enc), torch.cos(y_enc)), dim=-1)

        encoded_input = torch.cat((enc_x, enc_y), dim=-1)
        return encoded_input

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=16):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.frequency = self.generate_frequencies()

    def generate_frequencies(self):
        frequencies = torch.zeros(self.d_model // 2)
        frequencies[0] = 0 
        period_length = 72
        for i in range(1, self.d_model // 2):
            frequencies[i] = 2 * math.pi / period_length
            period_length /= 2
        return frequencies

    def forward(self, x):
        batch_size, _ = x.shape

        x_enc = x[:, :1] * self.frequency
        y_enc = x[:, 1:] * self.frequency
        enc_x = torch.cat((torch.sin(x_enc), torch.cos(x_enc)), dim=-1)
        enc_y = torch.cat((torch.sin(y_enc), torch.cos(y_enc)), dim=-1)

        encoded_input = torch.cat((enc_x, enc_y), dim=-1)
        return encoded_input

class INR(ABC, nn.Module):
    def __init__(self, seed, INR_model_config):
        super(INR, self).__init__()
        self.input_feature_dim = INR_model_config.get("input_feature_dim")
        self.output_feature_dim = INR_model_config.get("output_feature_dim")
        self.seed = seed
        self.INR_model_config = INR_model_config
        
        set_seed(self.seed)
        self.build_model()

    @abstractmethod
    def build_model(self):
        pass

class sMLP(INR):
    def build_model(self):
        hidden_features = self.INR_model_config.get('hidden_features', None)
        layers = self.INR_model_config.get('layers', None)
        positional = self.INR_model_config.get('positional', None)
        d_model = self.INR_model_config.get('d_model', None)

        self.positional = positional
        input_features = self.input_feature_dim

        if self.positional:
            self.positional_encoding = PositionalEncoding(d_model=d_model)
            input_features = d_model * 2

        self.layers = nn.ModuleList()
        self.layer_sizes = []
        current_features = input_features

        for i in range(layers):
            next_features = hidden_features if i < layers - 1 else self.output_feature_dim
            self.layers.append(nn.Linear(current_features, next_features))
            self.layer_sizes.append((current_features * next_features + next_features))
            current_features = next_features

        self.activations = nn.ModuleList([nn.LeakyReLU() for _ in range(layers)])

    def forward(self, x, external_parameters=None):
        if self.positional:
            x = self.positional_encoding(x)

        for i, (linear, activation) in enumerate(zip(self.layers, self.activations)):
            if external_parameters is not None:
                weight = external_parameters[f"layers.{i}.weight"]
                bias = external_parameters[f"layers.{i}.bias"]
                x = F.linear(x, weight, bias)
            else:
                x = linear(x)
            x = activation(x)

        return x

    def __str__(self):
        return "sMLP"