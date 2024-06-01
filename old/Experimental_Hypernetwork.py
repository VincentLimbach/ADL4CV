import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from nfn.common import network_spec_from_wsfeat, state_dict_to_tensors, params_to_state_dicts, WeightSpaceFeatures
from architectures import *
from torch.utils.data import default_collate
import json
from utils import *

import matplotlib.pyplot as plt

with open("config.json") as json_file:
    INR_model_config = json.load(json_file)["INR_model_config"]

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

first_image, _ = train_dataset[0]
second_image, _ = train_dataset[1]
concat_image = torch.cat([first_image, second_image], dim=2)

coords = [[i, j] for i in range(28) for j in range(56)]
coords = torch.tensor(coords, dtype=torch.float32)
intensities = concat_image.view(-1).float()

model_paths_MLP = ['data/INR/sMLP/sMLP_0.pth', 'data/INR/sMLP/sMLP_1.pth']

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

def test_nfn():
    nfn_channels = 32

    hypernet = HypernetNFN(nfn_channels, input_channels=2)
    hyper_optimizer = optim.Adam(hypernet.parameters(), lr=0.0001)
    new_model = sMLP(42, INR_model_config)
    layer_names = ["layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias"]

    epochs = 500
    for epoch in range(epochs):
        hyper_optimizer.zero_grad()
        predicted_weights = hypernet(wsfeat)
        external_parameters = unflatten_weights(predicted_weights, new_model)
        outputs = new_model(coords, external_parameters)
        loss = F.mse_loss(outputs.squeeze(), intensities)
        loss.backward()

        hyper_optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    predicted_image = outputs.view(28, 56).detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(concat_image[0], cmap='gray')
    axes[0].set_title("Original Concatenated Image")
    axes[0].axis('off')

    axes[1].imshow(predicted_image, cmap='gray')
    axes[1].set_title("Predicted Image")
    axes[1].axis('off')

    plt.show()

    return predicted_weights

test_nfn()