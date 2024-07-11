import json 
import torch
import torch.nn as nn

from ADL4CV.architectures import PositionalEncoding

"""class GroupLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, groups=8):
        super(GroupLinearLayer, self).__init__()
        self.groups = groups
        self.in_features = in_features
        self.out_features = out_features
        self.group_in_features = in_features // groups
        self.group_out_features = math.ceil(out_features / groups)

        assert in_features % groups == 0, "in_features must be divisible by groups"

        self.weight = nn.Parameter(torch.Tensor(groups, self.group_out_features, self.group_in_features))
        self.bias = nn.Parameter(torch.Tensor(groups, self.group_out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1, 1)
        
    def forward(self, x):
        batch_size, num_features = x.size()
        assert num_features == self.in_features, f"Expected input features {self.in_features}, but got {num_features}"

        # Reshape to (batch_size * groups, group_in_features)
        x = x.view(batch_size, self.groups, self.group_in_features)

        # Perform batched matrix multiplication and add bias
        out = torch.bmm(x, self.weight.permute(0, 2, 1)) + self.bias.unsqueeze(0).unsqueeze(0)

        # Reshape to (batch_size, groups * group_out_features)
        out = out.view(batch_size, -1)
        
        return out[:, :self.out_features]

class HyperNetworkTranslator(nn.Module):
    def __init__(self):
        super(HyperNetworkTranslator, self).__init__()
        
        with open("config.json") as file:
            hypernetwork_config = json.load(file)["Hypernetwork_config_2D"]
        
        self.d_model = 11
        self.inital_size = hypernetwork_config["INRs_size"] +  2 * self.d_model + 105
        self.fc1 = GroupLinearLayer(self.inital_size, 256)
        self.fc2 = GroupLinearLayer(self.inital_size + 256, 256)
        self.fc3 = GroupLinearLayer(self.inital_size + 512, 256)
        self.fc4 = GroupLinearLayer(self.inital_size + 768, hypernetwork_config["INRs_size"])
        
        self.relu = nn.ReLU()
        self.positional_encoding = PositionalEncoding(self.d_model)

        print(sum(p.numel() for p in self.parameters()))
        
    def forward(self, x_init, offsets):
        device = x_init.device
        encoded_offset = self.positional_encoding(offsets).to(device)
        x_init = torch.cat((x_init, encoded_offset), dim=1)
        zero_padding = torch.zeros(x_init.size(0), 107, device=device)
        x_init = torch.cat((x_init, zero_padding), dim=1)
        print(x_init.shape)
        x_1 = self.relu(self.fc1(x_init))
        x_2 = torch.cat((x_init, x_1), dim=1)
        x_2 = self.relu(self.fc2(x_2))
        x_3 = torch.cat((x_init, x_1, x_2), dim=1)
        x_3 = self.relu(self.fc3(x_3))
        x_4 = torch.cat((x_init, x_1, x_2, x_3), dim=1)
        x_out = self.fc4(x_4)
        
        return x_out
"""


class HyperNetworkTranslator(nn.Module):
    def __init__(self):
        super(HyperNetworkTranslator, self).__init__()
        
        with open("config.json") as file:
            hypernetwork_config = json.load(file)["Hypernetwork_config_2D"]
        
        self.d_model=12
        
        self.inital_size = hypernetwork_config["INRs_size"] + 2 * self.d_model
        
        self.fc1 = nn.Linear(self.inital_size, 256)
        self.fc2 = nn.Linear(self.inital_size + 256, 256)
        self.fc3 = nn.Linear(self.inital_size + 512, 256)
        self.fc4 = nn.Linear(self.inital_size + 768, hypernetwork_config["INRs_size"] )
        
        self.relu = nn.ReLU()

        self.positional_encoding = PositionalEncoding(self.d_model)

        print(sum(p.numel() for p in self.parameters()))

        
    def forward(self, x_init, offsets):
        device = x_init.device
        encoded_offset = self.positional_encoding(offsets).to(device)
        x_0 = torch.cat((x_init, encoded_offset), dim=1)
        x_1 = self.relu(self.fc1(x_0))
        x_2 = torch.cat((x_0, x_1), dim=1)
        x_2 = self.relu(self.fc2(x_2))
        x_3 = torch.cat((x_0, x_1, x_2), dim=1)
        x_3 = self.relu(self.fc3(x_3))
        x_4 = torch.cat((x_0, x_1, x_2, x_3), dim=1)
        x_out = self.fc4(x_4)
        
        return x_out

class HyperNetworkMerger(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(HyperNetworkMerger, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config_2D"]
        self.d_model=12
        self.shared_output_dim = 64
        self.class_embedding_dim=16
        self.positional_encoding = PositionalEncoding(self.d_model)

        self.shared_fc1 = nn.Linear(self.hypernetwork_config["INRs_size"] + self.d_model*2, 256)
        self.shared_relu1 = nn.ReLU()
        self.shared_dropout1 = nn.Dropout(dropout_rate)
        self.shared_fc2 = nn.Linear(256, self.shared_output_dim)
        self.shared_relu2 = nn.ReLU()
        self.shared_dropout2 = nn.Dropout(dropout_rate)

        self.label_embedding = nn.Embedding(10, self.class_embedding_dim)
        self.dim = self.hypernetwork_config["INRs_size"]*2 + self.d_model * 4 + self.shared_output_dim * 2 + self.class_embedding_dim * 2

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(self.dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(self.dim + 256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(self.dim + 2*256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(self.dim + 3*256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(self.dim + 4*256, self.hypernetwork_config["INRs_size"])

        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x, offsets, label_1_batch, label_2_batch):
        device = x.device
        
        encoded_offset = self.positional_encoding(offsets).to(device)

        xs_left = x[:, :x.size(1) // 2]    
        xs_right = x[:, x.size(1) // 2:]   
        xs_left = torch.cat((xs_left, encoded_offset[:, :encoded_offset.size(1) // 2]), dim=1)
        xs_right = torch.cat((xs_right, encoded_offset[:, encoded_offset.size(1) // 2:]), dim=1)

        xs_left = self.shared_relu1(self.shared_fc1(xs_left))
        xs_left = self.shared_dropout1(xs_left)
        xs_left = self.shared_relu2(self.shared_fc2(xs_left))
        xs_left = self.shared_dropout2(xs_left)
        
        xs_right = self.shared_relu1(self.shared_fc1(xs_right))
        xs_right = self.shared_dropout1(xs_right)
        xs_right = self.shared_relu2(self.shared_fc2(xs_right))
        xs_right = self.shared_dropout2(xs_right)

        label_1_embeddings = self.label_embedding(label_1_batch).to(device)
        label_2_embeddings = self.label_embedding(label_2_batch).to(device)

        x_init = torch.cat((x, encoded_offset, xs_left, xs_right, label_1_embeddings, label_2_embeddings), dim=1)
        
        x_1 = self.fc1(x_init)
        x_1 = self.bn1(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.dropout(x_1)
        
        x_2 = torch.cat((x_init, x_1), dim=1)
        x_2 = self.fc2(x_2)
        x_2 = self.bn2(x_2)
        x_2 = self.relu(x_2)
        x_2 = self.dropout(x_2)
        
        x_3 = torch.cat((x_init, x_1, x_2), dim=1)
        x_3 = self.fc3(x_3)
        x_3 = self.bn3(x_3)
        x_3 = self.relu(x_3)
        x_3 = self.dropout(x_3)
        
        x_4 = torch.cat((x_init, x_1, x_2, x_3), dim=1)
        x_4 = self.fc4(x_4)
        x_4 = self.bn4(x_4)
        x_4 = self.relu(x_4)
        x_4 = self.dropout(x_4)

        x = torch.cat((x_init, x_1, x_2, x_3, x_4), dim=1)
        x = self.fc5(x)
        
        return x
    
class HyperNetworkTrueRes(nn.Module):
    def __init__(self):
        super(HyperNetworkTrueRes, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config_2D"]
        self.d_model=12
        self.shared_output_dim = 64
        self.class_embedding_dim=16
        self.positional_encoding = PositionalEncoding(self.d_model)

        self.shared_fc1 = nn.Linear(self.hypernetwork_config["INRs_size"] + self.d_model*2, 256)
        self.shared_relu1 = nn.ReLU()
        self.shared_fc2 = nn.Linear(256, self.shared_output_dim)
        self.shared_relu2 = nn.ReLU()

        self.label_embedding = nn.Embedding(10, self.class_embedding_dim)
        self.dim = self.hypernetwork_config["INRs_size"]*2 + self.d_model * 4 + self.shared_output_dim * 2 + self.class_embedding_dim * 2

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.dim, 256)
        self.fc2 = nn.Linear(self.dim + 256, 256)
        self.fc3 = nn.Linear(self.dim + 2*256, 256)
        self.fc4 = nn.Linear(self.dim + 3*256, 256)
        self.fc5 = nn.Linear(self.dim + 4*256, self.hypernetwork_config["INRs_size"])

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
        
        x_1 = self.relu(self.fc1(x_init))
        x_2 = torch.cat((x_init, x_1), dim=1)
        x_2 = self.relu(self.fc2(x_2))
        x_3 = torch.cat((x_init, x_1, x_2), dim=1)
        x_3 = self.relu(self.fc3(x_3))
        x_4 = torch.cat((x_init, x_1, x_2, x_3), dim=1)
        x_4 = self.relu(self.fc4(x_4))
        x = torch.cat((x_init, x_1, x_2, x_3, x_4), dim=1)
        x = self.fc5(x)
        return x


class HyperNetwork3D(nn.Module):
    def __init__(self):
        super(HyperNetwork3D, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config_3D"]

        self.fc1 = nn.Linear(self.hypernetwork_config["input_dim"], 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 256)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(256, self.hypernetwork_config["output_dim"], bias=False)

        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.fc4(x)
    
class HyperNetwork3D2(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(HyperNetwork3D2, self).__init__()
        with open("config.json") as file:
            self.hypernetwork_config = json.load(file)["Hypernetwork_config_3D"]
        self.d_model=12
        self.shared_output_dim = 64
        self.class_embedding_dim=16
        self.positional_encoding = PositionalEncoding(self.d_model)

        self.label_embedding = nn.Embedding(10, self.class_embedding_dim)
        self.dim = self.hypernetwork_config["input_dim"]

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(self.dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(self.dim + 256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(self.dim + 2*256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(self.dim + 3*256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(4*256, self.hypernetwork_config["output_dim"])

        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        
        x_1 = self.fc1(x)
        x_1 = self.bn1(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.dropout(x_1)
        
        x_2 = torch.cat((x, x_1), dim=1)
        x_2 = self.fc2(x_2)
        x_2 = self.bn2(x_2)
        x_2 = self.relu(x_2)
        x_2 = self.dropout(x_2)
        
        x_3 = torch.cat((x, x_1, x_2), dim=1)
        x_3 = self.fc3(x_3)
        x_3 = self.bn3(x_3)
        x_3 = self.relu(x_3)
        x_3 = self.dropout(x_3)
        
        x_4 = torch.cat((x, x_1, x_2, x_3), dim=1)
        x_4 = self.fc4(x_4)
        x_4 = self.bn4(x_4)
        x_4 = self.relu(x_4)
        x_4 = self.dropout(x_4)

        x = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        x = self.fc5(x)
        
        return x
    
class SharpNet(nn.Module):
    def __init__(self):
        super(SharpNet, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.relu = nn.ReLU()
        self.W_org = nn.Parameter(torch.ones((72,72)).squeeze(0).to(device))
        self.conv1 = nn.Conv2d(1, 16, 3, padding="same")
        self.conv2 = nn.Conv2d(16, 32, 3, padding="same")
        self.conv3 = nn.Conv2d(32, 1, 3, padding="same")

        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x_org):
        x = x_org.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x_sharp = self.relu(self.conv3(x))
        return torch.clip(x_org * self.W_org + x_sharp.squeeze(1) * (1-self.W_org), min =0, max=1)

