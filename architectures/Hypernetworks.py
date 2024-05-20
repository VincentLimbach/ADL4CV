import json 
import torch
import torch.nn as nn

from nfn import layers
from nfn.common import network_spec_from_wsfeat
from architectures import Autoencoder

class MergedHyperNetwork(nn.Module):
    def __init__(self):
        super(MergedHyperNetwork, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config"]

        self.fc1 = nn.Linear(self.hypernetwork_config["input_dim"], 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()

        self.autoencoder = Autoencoder.Autoencoder()
        self.autoencoder.load_state_dict(torch.load("models/autoencoder/autoencoder.pth"))
        self.autoencoder.eval()

        self.fc3 = nn.Linear(512, 512)  # 256 (fc2) + 256 (latent representation) = 512
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, self.hypernetwork_config["output_dim"])

    def forward(self, x):
        # Pass through MLP part
        x_mlp = self.relu1(self.fc1(x))
        x_mlp = self.relu2(self.fc2(x_mlp))
        
        # Get latent representations
        half_size = x.size(1) // 2
        x1 = x[:, :half_size]
        x2 = x[:, half_size:]
        
        with torch.no_grad():
            latent1 = self.autoencoder.encoder(x1)
            latent2 = self.autoencoder.encoder(x2)
        
        # Concatenate the latent representations with MLP output
        latent_concat = torch.cat((latent1, latent2), dim=1)
        combined = torch.cat((x_mlp, latent_concat), dim=1)
        
        # Pass through final layers
        x = self.relu3(self.fc3(combined))
        x = self.fc4(x)
        
        return x

class HyperNetworkIFE(nn.Module):
    def __init__(self):
        super(HyperNetworkIFE, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config"]
        self.autoencoder = Autoencoder.Autoencoder()
        self.autoencoder.load_state_dict(torch.load("models/autoencoder/autoencoder.pth"))
        self.fc1 = nn.Linear(512, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, self.hypernetwork_config["output_dim"])
        self.relu2 = nn.ReLU()

    def forward(self, x):
        half_size = x.size(1) // 2
        x1 = x[:, :half_size]
        x2 = x[:, half_size:]
        
        latent1 = self.autoencoder.encoder(x1)
        latent2 = self.autoencoder.encoder(x2)
        
        latent_concat = torch.cat((latent1, latent2), dim=1)
        
        x = self.relu1(self.fc1(latent_concat))
        x = self.relu2(self.fc2(x))
        
        return x
        

class HyperNetworkMLP(nn.Module):
    def __init__(self):
        super(HyperNetworkMLP, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config"]

        self.fc1 = nn.Linear(self.hypernetwork_config["input_dim"], 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, self.hypernetwork_config["output_dim"], bias=False)
        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
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