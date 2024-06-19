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
        self.fc3 = nn.Linear(256 + spectral_encoding_dim*2, 512)
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

import torch

def spectral_encoding(labels, spectral_encoding_dim=8):
    labels = labels.float().unsqueeze(1) 
    frequencies = torch.arange(spectral_encoding_dim, device=labels.device).float().unsqueeze(0)
    encoding = torch.sin(labels * (frequencies / spectral_encoding_dim))
    return encoding

class HyperNetworkMLPGeneralExtendedStable(nn.Module):
    def __init__(self, spectral_encoding_dim=8):
        super(HyperNetworkMLPGeneralExtended, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config"]

        self.positional_encoding = PositionalEncoding(d_model=8)
        self.spectral_encoding_dim = spectral_encoding_dim

        self.shared_fc1 = nn.Linear(self.hypernetwork_config["input_dim"]//2+16, 256)
        self.shared_relu1 = nn.ReLU()
        self.shared_fc2 = nn.Linear(256, 64)
        self.shared_relu2 = nn.ReLU()

        self.fc1 = nn.Linear(self.hypernetwork_config["input_dim"] + 32, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(384, 384)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(384, 384)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(384, 512)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(512, self.hypernetwork_config["output_dim"])
        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x, offsets, label_1_batch, label_2_batch):
        device = x.device
        
        offsets = offsets.to(device)
        
        encoded_offset = self.positional_encoding(offsets).to(device)
        size = x.size(0)
        xs_left = x[:,:x.size(1)//2]    
        xs_right = x[:,x.size(1)//2:]   

        x = torch.cat((x, encoded_offset), dim=1)
        
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        
        xs_left = torch.cat((xs_left, encoded_offset[:,:encoded_offset.size(1)//2]), dim=1)
        xs_right = torch.cat((xs_right, encoded_offset[:,encoded_offset.size(1)//2:]), dim=1)

        xs_left = self.shared_relu1(self.shared_fc1(xs_left))
        xs_left = self.shared_relu2(self.shared_fc2(xs_left))
        
        xs_right = self.shared_relu1(self.shared_fc1(xs_right))
        xs_right = self.shared_relu2(self.shared_fc2(xs_right))

        x = torch.cat((x, xs_left, xs_right), dim=1)
        
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.relu5(self.fc5(x))
        x = self.fc6(x)
        return x


class HyperNetworkMLPGeneralExtended(nn.Module):
    def __init__(self, spectral_encoding_dim=8):
        super(HyperNetworkMLPGeneralExtended, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config"]

        self.positional_encoding = PositionalEncoding(d_model=8)
        self.spectral_encoding_dim = spectral_encoding_dim

        self.shared_fc1 = nn.Linear(self.hypernetwork_config["input_dim"] // 2 + 16, 256)
        self.shared_relu1 = nn.ReLU()
        self.shared_fc2 = nn.Linear(256, 64)
        self.shared_relu2 = nn.ReLU()

        self.label_embedding = nn.Embedding(10, 16)  # Assuming 10 classes

        self.fc1 = nn.Linear(self.hypernetwork_config["input_dim"] + 32, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(416, 416)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(416, 256)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(416, 416)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(416, 416)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(416, 512)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(512, self.hypernetwork_config["output_dim"])

        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x, offsets, label_1_batch, label_2_batch):
        device = x.device
        
        encoded_offset = self.positional_encoding(offsets).to(device)
        xs_left = x[:, :x.size(1) // 2]    
        xs_right = x[:, x.size(1) // 2:]   

        x = torch.cat((x, encoded_offset), dim=1)
        
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        
        xs_left = torch.cat((xs_left, encoded_offset[:, :encoded_offset.size(1) // 2]), dim=1)
        xs_right = torch.cat((xs_right, encoded_offset[:, encoded_offset.size(1) // 2:]), dim=1)

        xs_left = self.shared_relu1(self.shared_fc1(xs_left))
        xs_left = self.shared_relu2(self.shared_fc2(xs_left))
        
        xs_right = self.shared_relu1(self.shared_fc1(xs_right))
        xs_right = self.shared_relu2(self.shared_fc2(xs_right))

        label_1_embeddings = self.label_embedding(label_1_batch).to(device)
        label_2_embeddings = self.label_embedding(label_2_batch).to(device)

        x = torch.cat((x, xs_left, xs_right, label_1_embeddings, label_2_embeddings), dim=1)
        
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = torch.cat((x, xs_left, xs_right, label_1_embeddings, label_2_embeddings), dim=1)

        x = self.relu5(self.fc5(x))
        x = self.relu6(self.fc6(x))
        x = self.relu7(self.fc7(x))
        x = self.fc8(x)
        return x

class HyperNetworkRes(nn.Module):
    def __init__(self):
        super(HyperNetworkRes, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config"]
        self.d_model=12
        self.shared_output_dim = 64
        self.class_embedding_dim=16
        self.positional_encoding = PositionalEncoding(self.d_model)

        self.shared_fc1 = nn.Linear(self.hypernetwork_config["input_dim"] // 2 + self.d_model*2, 256)
        self.shared_relu1 = nn.ReLU()
        self.shared_fc2 = nn.Linear(256, self.shared_output_dim)
        self.shared_relu2 = nn.ReLU()

        self.label_embedding = nn.Embedding(10, self.class_embedding_dim)
        self.dim = self.hypernetwork_config["input_dim"] + self.d_model * 4 + self.shared_output_dim * 2 + self.class_embedding_dim * 2
        
        self.fc1 = nn.Linear(self.dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.dim +256, 256)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(self.dim +256, 256)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(self.dim +256, 512)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(self.dim + 512, self.hypernetwork_config["output_dim"])

        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x, offsets, label_1_batch, label_2_batch):
        device = x.device
        
        encoded_offset = self.positional_encoding(offsets).to(device)

        xs_left = x[:, :x.size(1) // 2]    
        xs_right = x[:, x.size(1) // 2:]   
        xs_left = torch.cat((xs_left, encoded_offset[:, :encoded_offset.size(1) // 2]), dim=1)
        xs_right = torch.cat((xs_right, encoded_offset[:, encoded_offset.size(1) // 2:]), dim=1)

        xs_left = self.shared_relu1(self.shared_fc1(xs_left))
        xs_left = self.shared_relu2(self.shared_fc2(xs_left))
        
        xs_right = self.shared_relu1(self.shared_fc1(xs_right))
        xs_right = self.shared_relu2(self.shared_fc2(xs_right))

        label_1_embeddings = self.label_embedding(label_1_batch).to(device)
        label_2_embeddings = self.label_embedding(label_2_batch).to(device)


        x_init = torch.cat((x, encoded_offset, xs_left, xs_right, label_1_embeddings, label_2_embeddings), dim=1)
        
        x = self.relu1(self.fc1(x_init))
        x = torch.cat((x, x_init), dim=1)
        x = self.relu2(self.fc2(x))
        x = torch.cat((x, x_init), dim=1)
        x = self.relu3(self.fc3(x))
        x = torch.cat((x, x_init), dim=1)
        x = self.relu4(self.fc4(x))
        x = torch.cat((x, x_init), dim=1)
        x = self.fc5(x)
        return x

class HyperNetworkTrueRes(nn.Module):
    def __init__(self):
        super(HyperNetworkTrueRes, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config"]
        self.d_model=12
        self.shared_output_dim = 64
        self.class_embedding_dim=16
        self.positional_encoding = PositionalEncoding(self.d_model)

        self.shared_fc1 = nn.Linear(self.hypernetwork_config["input_dim"] // 2 + self.d_model*2, 256)
        self.shared_relu1 = nn.ReLU()
        self.shared_fc2 = nn.Linear(256, self.shared_output_dim)
        self.shared_relu2 = nn.ReLU()

        self.label_embedding = nn.Embedding(10, self.class_embedding_dim)
        self.dim = self.hypernetwork_config["input_dim"] + self.d_model * 4 + self.shared_output_dim * 2 + self.class_embedding_dim * 2
        
        self.fc1 = nn.Linear(self.dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.dim + 256, 256)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(self.dim + 2*256, 256)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(self.dim + 3*256, 256)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(self.dim + 4*256, self.hypernetwork_config["output_dim"])

        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x, offsets, label_1_batch, label_2_batch):
        device = x.device
        
        encoded_offset = self.positional_encoding(offsets).to(device)

        xs_left = x[:, :x.size(1) // 2]    
        xs_right = x[:, x.size(1) // 2:]   
        xs_left = torch.cat((xs_left, encoded_offset[:, :encoded_offset.size(1) // 2]), dim=1)
        xs_right = torch.cat((xs_right, encoded_offset[:, encoded_offset.size(1) // 2:]), dim=1)

        xs_left = self.shared_relu1(self.shared_fc1(xs_left))
        xs_left = self.shared_relu2(self.shared_fc2(xs_left))
        
        xs_right = self.shared_relu1(self.shared_fc1(xs_right))
        xs_right = self.shared_relu2(self.shared_fc2(xs_right))

        label_1_embeddings = self.label_embedding(label_1_batch).to(device)
        label_2_embeddings = self.label_embedding(label_2_batch).to(device)

        x_init = torch.cat((x, encoded_offset, xs_left, xs_right, label_1_embeddings, label_2_embeddings), dim=1)
        
        x_1 = self.relu1(self.fc1(x_init))
        x_2 = torch.cat((x_init, x_1), dim=1)
        x_2 = self.relu2(self.fc2(x_2))
        x_3 = torch.cat((x_init, x_1, x_2), dim=1)
        x_3 = self.relu3(self.fc3(x_3))
        x_4 = torch.cat((x_init, x_1, x_2, x_3), dim=1)
        x_4 = self.relu4(self.fc4(x_4))
        x = torch.cat((x_init, x_1, x_2, x_3, x_4), dim=1)
        x = self.fc5(x)
        return x
    

class HyperNetworkTrueResBig(nn.Module):
    def __init__(self):
        super(HyperNetworkTrueResBig, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config"]
        self.d_model = 12
        self.shared_output_dim = 64
        self.class_embedding_dim = 16
        self.positional_encoding = PositionalEncoding(self.d_model)

        self.shared_fc1 = nn.Linear(self.hypernetwork_config["input_dim"] // 2 + self.d_model*2, 256)
        self.shared_relu1 = nn.ReLU()
        self.shared_fc2 = nn.Linear(256, self.shared_output_dim)
        self.shared_relu2 = nn.ReLU()

        self.label_embedding = nn.Embedding(10, self.class_embedding_dim)
        self.dim = self.hypernetwork_config["input_dim"] + self.d_model * 4 + self.shared_output_dim * 2 + self.class_embedding_dim * 2

        # Define the ResNet-like blocks
        self.res_blocks = nn.ModuleList([self._build_res_block(i) for i in range(7)])

        self.fc_out = nn.Linear(self.dim + 7 * 256, self.hypernetwork_config["output_dim"])

        print(sum(p.numel() for p in self.parameters()))

    def _build_res_block(self, i):
        dim = self.dim + 256 * i
        layers = []
        layers.append(nn.Linear(dim, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 256))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x, offsets, label_1_batch, label_2_batch):
        device = x.device

        encoded_offset = self.positional_encoding(offsets).to(device)

        xs_left = x[:, :x.size(1) // 2]
        xs_right = x[:, x.size(1) // 2:]
        xs_left = torch.cat((xs_left, encoded_offset[:, :encoded_offset.size(1) // 2]), dim=1)
        xs_right = torch.cat((xs_right, encoded_offset[:, encoded_offset.size(1) // 2:]), dim=1)

        xs_left = self.shared_relu1(self.shared_fc1(xs_left))
        xs_left = self.shared_relu2(self.shared_fc2(xs_left))

        xs_right = self.shared_relu1(self.shared_fc1(xs_right))
        xs_right = self.shared_relu2(self.shared_fc2(xs_right))

        label_1_embeddings = self.label_embedding(label_1_batch).to(device)
        label_2_embeddings = self.label_embedding(label_2_batch).to(device)

        x_init = torch.cat((x, encoded_offset, xs_left, xs_right, label_1_embeddings, label_2_embeddings), dim=1)

        x_res = x_init
        for res_block in self.res_blocks:
            x_res = torch.cat((x_res, res_block(x_res)), dim=1)
        x = self.fc_out(x_res)
        return x