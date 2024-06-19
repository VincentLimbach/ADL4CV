from INRTrainer import INRTrainer
from torchvision import datasets, transforms
from architectures import sMLP
import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
from HypernetworkTrainer import *
import json

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainer = INRTrainer(debug=True)
#trainer.fit_inrs(dataset, np.arange(100), sMLP, 42, "./data/INR/sMLP_reg/sMLP_5-1e-4_300_")


def flattened_params(model):
    n = sum(p.numel() for p in model.parameters())
    params = torch.zeros(n)
    i = 0
    for p in model.parameters():
        params_slice = params[i:i + p.numel()]
        params_slice.copy_(p.flatten())
        p.data = params_slice.view(p.shape)
        i += p.numel()
    return params.detach().numpy()

with open("config.json") as json_file:
    INR_model_config = json.load(json_file)["INR_model_config"]
model = sMLP(seed=42, INR_model_config=INR_model_config)

params = []

k_lst = [25, 50, 100, 250, 300, 500, 1000, 2500, 5000, 10000]

for k in k_lst:
    k_params = []
    for i in range(100):
        model.load_state_dict(torch.load(f"./data/INR/sMLP_comparison/sMLP{k}_{i}.pth"))
        k_params.append(flattened_params(model))
    params.append(k_params)

params_reg = []

k_lst = [25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

for k in k_lst:
    k_params = []
    for i in range(100):
        model.load_state_dict(torch.load(f"./data/INR/sMLP_reg/sMLP_5-1e-4_{k}_{i}.pth"))
        k_params.append(flattened_params(model))
    params_reg.append(k_params)

euclidian_distances = []

for k in range(len(k_lst)):
    k_dist = []
    for i in range(100):
        for j in range(i,100):
            if i == j:
                continue
            k_dist.append(np.linalg.norm(params[k][i] - params[k][j]))
    euclidian_distances.append(k_dist)

euclidian_distances_reg = []

for k in range(len(k_lst)):
    k_dist = []
    for i in range(100):
        for j in range(i,100):
            if i == j:
                continue
            k_dist.append(np.linalg.norm(params_reg[k][i] - params_reg[k][j]))
    euclidian_distances_reg.append(k_dist)

euclidian_distances = np.array(euclidian_distances)
euclidian_distances_reg = np.array(euclidian_distances_reg)


plt.plot(k_lst, np.mean(euclidian_distances, axis=1), label = "unreg")
plt.plot(k_lst, np.mean(euclidian_distances_reg, axis=1), label = "reg")
plt.xscale("log")
plt.xticks(k_lst)
ax = plt.gca()
plt.xlabel("Epochs")
plt.ylabel("Average euclidian distance")
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.show()


