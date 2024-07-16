import json
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

from ADL4CV.architectures import HyperNetworkTrueRes, sMLP
from ADL4CV.data import *
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_INRpair2D(
    dataset, pair_index, offsets, model_cls, seed, save=False, save_name=None
):
    img_1, _ = dataset[pair_index[0]]
    img_2, _ = dataset[pair_index[1]]

    img_concat = generate_merged_image(
        img_1, img_2, offsets[:, 0], offsets[:, 1], offsets[:, 2], offsets[:, 3], device
    )
    height, width = img_concat.shape[1], img_concat.shape[2]

    coords = [[i, j] for i in range(height) for j in range(width)]
    intensities = [
        img_concat[:, i, j].item() for i in range(height) for j in range(width)
    ]
    coords = torch.tensor(coords, dtype=torch.float32)
    intensities = torch.tensor(intensities, dtype=torch.float32)

    model = model_cls(seed=seed, INR_model_config=INR_model_config)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=INR_trainer_config["lr"],
        weight_decay=INR_trainer_config.get("weight_decay", 0),
    )

    losses = []

    t0 = time.time()
    epochs = INR_trainer_config["epochs"]
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(coords)
        start_time = time.time()
        loss = criterion(outputs.squeeze(), intensities)
        losses.append(loss.detach().item())
        loss.backward()
        start_time = time.time()
        optimizer.step()
        start_time = time.time()

        if epoch % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    if save:
        torch.save(model.state_dict(), save_name)

    return losses


with open("config.json") as json_file:
    json_file = json.load(json_file)
    INR_model_config = json_file["INR_model_config_2D"]
    INR_trainer_config = json_file["INR_trainer_config_2D"]

train_pairs, weak_val_pairs = generate_pairs(8192, 256, 32, seed=42)
strong_val_pairs, _ = generate_pairs(256, 0, 1, seed=42)
strong_val_pairs = [(i + 8192, j + 8192) for i, j in strong_val_pairs]

torch.manual_seed(42)
offsets_train = torch.randint(1, 16, (50, 4), device=device) * 2
offsets_weak_val = torch.randint(1, 16, (50, 4), device=device) * 2
offsets_strong_val = torch.randint(1, 16, (50, 4), device=device) * 2

dataset = datasets.MNIST(
    root="ADL4CV/data/gt_data/",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)


def INRpair_loss(pairs, offsets):
    losses = []
    for i in range(50):
        print(i)
        loss = train_INRpair2D(dataset, pairs[i], offsets[i].unsqueeze(0), sMLP, 42)
        losses.append(loss)
    return losses


# val_losses = INRpair_loss(weak_val_pairs[:2], offsets=offsets_weak_val)
# np.savetxt('ADL4CV/evaluation/weak_val_losses.txt', np.array(val_losses), delimiter=',')

train_losses = np.loadtxt("ADL4CV/evaluation/weak_val_losses.txt", delimiter=",")

ci = 1.96 * np.std(train_losses, axis=0) / np.sqrt(50)

plt.rcParams["figure.figsize"] = (4, 3)  # set figure size
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 12

x = np.arange(10000)
mean = np.mean(train_losses, axis=0)


# --------------- regular scaling
plt.plot(x, mean)
plt.fill_between(x, (mean - ci), (mean + ci), color="b", alpha=0.1)

threshold = 0.0043
loss_intercept = np.min(np.where(mean < threshold)[0])
print(loss_intercept)
y_min, y_max = plt.ylim()
x_min, x_max = plt.xlim()

plt.vlines(loss_intercept, ymax=threshold, ymin=0, linestyle="--", color="red")
plt.hlines(threshold, xmax=loss_intercept, xmin=0, linestyle="--", color="green")
plt.hlines(threshold, xmax=x_max, xmin=loss_intercept, linestyle="--", color="grey")
plt.xlabel("Epochs")
plt.ylabel("Mean training loss")
plt.savefig("./ADL4CV/evaluation/Mean_training_loss_regular.png", bbox_inches="tight")
# --------------- log-log
plt.plot(x, mean)
plt.fill_between(x, (mean - ci), (mean + ci), color="b", alpha=0.1)
plt.yscale("log")
plt.xscale("log")

threshold = 0.005
loss_intercept = np.min(np.where(mean < threshold)[0])
y_min, y_max = plt.ylim()
x_min, x_max = plt.xlim()

print(
    (np.log10(y_max) + np.log10(y_min)),
    10 ** ((np.log10(y_max) + np.log10(y_min)) * np.log10(threshold)),
)
plt.vlines(loss_intercept, ymax=threshold, ymin=0, linestyle="--", color="red")
plt.hlines(threshold, xmax=loss_intercept, xmin=0, linestyle="--", color="green")
plt.hlines(threshold, xmax=x_max, xmin=loss_intercept, linestyle="--", color="grey")
plt.xlabel("Epochs")
plt.ylabel("Mean training loss")
plt.savefig("./ADL4CV/evaluation/Mean_training_loss_loglog.png", bbox_inches="tight")
