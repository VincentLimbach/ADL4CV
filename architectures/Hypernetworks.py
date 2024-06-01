import json 
import torch
import torch.nn as nn
from architectures import PositionalEncoding

from architectures import Autoencoder

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
        
class HyperNetworkMLPConcat(nn.Module):
    def __init__(self):
        super(HyperNetworkMLPConcat, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config"]

        self.fc1 = nn.Linear(self.hypernetwork_config["input_dim"], 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, self.hypernetwork_config["output_dim"], bias=False)
        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

class HyperNetworkMLPGeneral(nn.Module):
    def __init__(self):
        super(HyperNetworkMLPGeneral, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config"]

        self.positional_encoding = PositionalEncoding(d_model=8)
        self.fc1 = nn.Linear(self.hypernetwork_config["input_dim"] + 32, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, self.hypernetwork_config["output_dim"], bias=False)
        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x, offsets):
        device = x.device
        
        offsets = offsets.to(device)
        
        encoded_offset = self.positional_encoding(offsets).to(device)
        
        x = torch.cat((x, encoded_offset), dim=1)
        
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x