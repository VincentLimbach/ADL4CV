import json 
import torch
import torch.nn as nn
from architectures import sMLP
from nfn import layers
from nfn.common import network_spec_from_wsfeat
from nfn.common import network_spec_from_wsfeat, state_dict_to_tensors, params_to_state_dicts, WeightSpaceFeatures

from torch.utils.data import default_collate

def unflatten_weights(flat_weights, model):
    external_parameters_list = []
    batch_size = flat_weights.size(0)
    
    for batch_idx in range(batch_size):
        external_parameters = {}
        offset = 0
        for i, layer in enumerate(model.layers):
            weight_shape = layer.weight.shape
            bias_shape = layer.bias.shape
            weight_numel = layer.weight.numel()
            bias_numel = layer.bias.numel()

            external_parameters[f"layers.{i}.weight"] = flat_weights[batch_idx, offset:offset + weight_numel].view(weight_shape)
            offset += weight_numel
            external_parameters[f"layers.{i}.bias"] = flat_weights[batch_idx, offset:offset + bias_numel].view(bias_shape)
            offset += bias_numel
        
        external_parameters_list.append(external_parameters)
    
    return external_parameters_list

with open("config.json") as json_file:
    INR_model_config = json.load(json_file)["INR_model_config"]


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

def extract_wsfeat(model_paths):
    models = []
    for path in model_paths:
        model = sMLP(42, INR_model_config)
        model.load_state_dict(torch.load(path))
        models.append(model)

    state_dicts = [m.state_dict() for m in models]  # Extract model params
    wts_and_bs = [state_dict_to_tensors(sd) for sd in state_dicts]  # Dict -> tensors
    wts_and_bs = default_collate(wts_and_bs)  # Process

    weights, biases = wts_and_bs
    weights_mod = []
    biases_mod = []
    for i in range(len(weights)):  # Input shape: [2,1,weights.shape] -> output shape: [1,2,weights.shape]
        weights_mod.append(torch.unsqueeze(torch.cat([weights[i][j] for j in range(len(models))]), dim=0))
        biases_mod.append(torch.unsqueeze(torch.cat([biases[i][j] for j in range(len(models))]), dim=0))

    wts_and_bs = [weights_mod, biases_mod]
    wsfeat = WeightSpaceFeatures(*wts_and_bs)

    return wsfeat


class HyperNetworkNFN(nn.Module):
    def __init__(self, nfn_channels=64, input_channels=2):
        super(HyperNetworkNFN, self).__init__()
        self.nfn_channels = nfn_channels
        self.input_channels = input_channels
        
        wsfeat = extract_wsfeat(['data/INR/sMLP/sMLP_0.pth', 'data/INR/sMLP/sMLP_1.pth'])
        network_spec = network_spec_from_wsfeat(wsfeat)

        self.network_spec = network_spec

        self.NP1 = layers.NPLinear(network_spec, self.input_channels, nfn_channels, io_embed=True)
        self.relu1 = layers.TupleOp(nn.ReLU())
        self.NP2 = layers.NPLinear(network_spec, nfn_channels, nfn_channels, io_embed=True)
        self.relu2 = layers.TupleOp(nn.ReLU())
        self.NP3 = layers.NPLinear(network_spec, nfn_channels, 1, io_embed=True)

        self.flatten = nn.Flatten(start_dim=0) 
        self.fc1 = nn.Linear(2177, 512)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(512, 2177)
        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        batch_size = x.size(0)
        split_idx = x.shape[1] // 2
        left_weights = x[:, :split_idx]
        right_weights = x[:, split_idx:]
        
        left_models = [sMLP(seed=42, INR_model_config=INR_model_config) for _ in range(batch_size)]
        right_models = [sMLP(seed=42, INR_model_config=INR_model_config) for _ in range(batch_size)]
        
        left_state_dicts = unflatten_weights(left_weights, left_models[0])
        right_state_dicts = unflatten_weights(right_weights, right_models[0])
        
        outputs = []
        for i in range(batch_size):
            left_models[i].load_state_dict(left_state_dicts[i])
            right_models[i].load_state_dict(right_state_dicts[i])
        
            left_model, right_model = left_models[i], right_models[i]
            wts_and_bs = [state_dict_to_tensors(left_model.state_dict()), state_dict_to_tensors(right_model.state_dict())]
            wts_and_bs = default_collate(wts_and_bs)

            weights, biases = wts_and_bs
            weights_mod = []
            biases_mod = []
            for j in range(len(weights)):
                weights_mod.append(torch.unsqueeze(torch.cat([weights[j][k] for k in range(2)]), dim=0))
                biases_mod.append(torch.unsqueeze(torch.cat([biases[j][k] for k in range(2)]), dim=0))

            wts_and_bs = [weights_mod, biases_mod]
            x_ws = WeightSpaceFeatures(*wts_and_bs)

            x_ws = self.NP1(x_ws)
            x_ws = self.relu1(x_ws)
            x_ws = self.NP2(x_ws)
            x_ws = self.relu2(x_ws)
            x_ws = self.NP3(x_ws)

            w0, w1 = x_ws.weights
            b0, b1 = x_ws.biases

            w0_flat = self.flatten(w0)
            w1_flat = self.flatten(w1)
            b0_flat = self.flatten(b0)
            b1_flat = self.flatten(b1)
            combined = torch.cat([w0_flat, w1_flat, b0_flat, b1_flat], dim=0)

            combined = self.relu_fc(self.fc1(combined))
            output = self.fc2(combined)
            outputs.append(output)
        
        return torch.stack(outputs, dim=0)
