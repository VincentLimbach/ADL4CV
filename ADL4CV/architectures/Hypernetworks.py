import json 
import torch
import torch.nn as nn

from ADL4CV.architectures import PositionalEncoding

class HyperNetworkTrueRes(nn.Module):
    def __init__(self):
        super(HyperNetworkTrueRes, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config_2D"]
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
    

class HyperNetwork3D(nn.Module):
    def __init__(self):
        super(HyperNetwork3D, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config_3D"]

        self.fc1 = nn.Linear(self.hypernetwork_config["input_dim"], 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
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