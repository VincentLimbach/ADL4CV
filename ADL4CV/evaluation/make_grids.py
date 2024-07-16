import json
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import torchvision.transforms.functional as F

from ADL4CV.architectures import HyperNetworkTrueRes, sMLP, HyperNetwork2D, SharpNet
from ADL4CV.data import *
from HypernetworkTrainer2D import *
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
with open("config.json") as file:
    new_model = sMLP(
        seed=42, INR_model_config=json.load(file)["INR_model_config_2D"], device=device
    )


inr_trainer = INRTrainer2D()
hypernetwork = HyperNetwork2D().to(device)
sharpnet = SharpNet().to(device)

hypernetwork_trainer = HyperNetworkTrainer(
    sharpnet,
    hypernetwork,
    sMLP,
    inr_trainer,
    save_path="ADL4CV/models/2D/hypernetwork_true_res.pth",
    load=True,
)
train_pairs, weak_val_pairs = generate_pairs(4096, 256, 8, seed=42)
strong_val_pairs, _ = generate_pairs(256, 0, 1, seed=42)
strong_val_pairs = [(i + 8192, j + 8192) for i, j in strong_val_pairs]

ncols = 6

dataset = ImageINRDataset(
    "MNIST",
    sMLP,
    inr_trainer,
    "ADL4CV/data/model_data/MNIST",
    on_the_fly=False,
    device=device,
)
weak_val_dataset = PairedDataset(dataset, weak_val_pairs[: ncols**2])
strong_val_dataset = PairedDataset(dataset, strong_val_pairs[: ncols**2])
weak_val_loader = DataLoader(weak_val_dataset, batch_size=64, shuffle=False)
strong_val_loader = DataLoader(strong_val_dataset, batch_size=64, shuffle=False)

_, weak_val_results = hypernetwork_trainer.validate(
    weak_val_loader, nn.MSELoss(), 0, debug=True, final=True, batch_results=True
)
_, strong_val_results = hypernetwork_trainer.validate(
    strong_val_loader, nn.MSELoss(), 0, debug=True, final=True, batch_results=True
)

gt_weak, out_weak = weak_val_results
gt_strong, out_strong = strong_val_results

plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.figsize"] = (10, 10)

plt.imshow(
    make_grid(gt_weak.unsqueeze(1), nrow=ncols)
    .permute(1, 2, 0)
    .clip(0, 1)
    .detach()
    .numpy()
)
plt.axis("off")
plt.savefig(
    f"./ADL4CV/evaluation/grids/GT_weak_{ncols}x{ncols}.png", bbox_inches="tight"
)

plt.imshow(
    make_grid(out_weak.unsqueeze(1), nrow=ncols)
    .permute(1, 2, 0)
    .clip(0, 1)
    .detach()
    .numpy()
)
plt.axis("off")
plt.savefig(
    f"./ADL4CV/evaluation/grids/Out_weak_{ncols}x{ncols}.png", bbox_inches="tight"
)

plt.imshow(
    make_grid(gt_strong.unsqueeze(1), nrow=ncols)
    .permute(1, 2, 0)
    .clip(0, 1)
    .detach()
    .numpy()
)
plt.axis("off")
plt.savefig(
    f"./ADL4CV/evaluation/grids/GT_strong_{ncols}x{ncols}.png", bbox_inches="tight"
)

plt.imshow(
    make_grid(out_strong.unsqueeze(1), nrow=ncols)
    .permute(1, 2, 0)
    .clip(0, 1)
    .detach()
    .numpy()
)
plt.axis("off")
plt.savefig(
    f"./ADL4CV/evaluation/grids/Out_strong_{ncols}x{ncols}.png", bbox_inches="tight"
)
