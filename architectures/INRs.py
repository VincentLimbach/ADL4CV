import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math 

from abc import ABC, abstractmethod

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=16, device=torch.device('cpu')):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.device = device
        self.frequency = self.generate_frequencies().to(self.device)

    def generate_frequencies(self):
        frequencies = torch.zeros(self.d_model // 2)
        frequencies[0] = 0 
        period_length = 72
        for i in range(1, self.d_model // 2):
            frequencies[i] = 2 * math.pi / period_length
            period_length /= 2
        return frequencies

    def forward(self, x):
        x = x.to(self.device)
        encodings = []
        for i in range(x.shape[1]):
            x_enc = x[:, i:i+1] * self.frequency
            enc_x = torch.cat((torch.sin(x_enc), torch.cos(x_enc)), dim=-1)
            encodings.append(enc_x)
        encoded_input = torch.cat(encodings, dim=-1)
        return encoded_input

class INR(ABC, nn.Module):
    def __init__(self, seed, INR_model_config, device=torch.device('cpu')):
        super(INR, self).__init__()
        self.input_feature_dim = INR_model_config.get("input_feature_dim")
        self.output_feature_dim = INR_model_config.get("output_feature_dim")
        self.seed = seed
        self.INR_model_config = INR_model_config
        self.device = device
        
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
            self.positional_encoding = PositionalEncoding(d_model=d_model, device=self.device)
            input_features = d_model * 2

        self.layers = nn.ModuleList()
        self.layer_sizes = []
        current_features = input_features

        for i in range(layers):
            next_features = hidden_features if i < layers - 1 else self.output_feature_dim
            self.layers.append(nn.Linear(current_features, next_features).to(self.device))
            self.layer_sizes.append((current_features * next_features + next_features))
            current_features = next_features

        self.activations = nn.ModuleList([nn.LeakyReLU().to(self.device) for _ in range(layers)])

    def forward(self, x, weights=None, biases=None):
        if self.positional:
            x = self.positional_encoding(x)
        
        for i, (linear, activation) in enumerate(zip(self.layers, self.activations)):
            if weights is not None and biases is not None:
                weight = weights[i]
                bias = biases[i]
                if x.ndim == 2:
                    x = x.unsqueeze(0).expand(weight.size(0), -1, -1)
                x = torch.bmm(x, weight.transpose(1, 2)) + bias.unsqueeze(1)
            else:
                x = linear(x)
            x = activation(x)
        return x


    def __str__(self):
        return "sMLP"