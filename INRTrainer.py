import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import datasets, transforms
#from dotenv import load_dotenv
from architectures import INR, sMLP
#from path import Path

class INRTrainer:
    def __init__(self, debug=False):
        self.debug = debug
        with open("config.json") as json_file:
            json_file = json.load(json_file)
            self.INR_model_config = json_file["INR_model_config"]
            self.INR_trainer_config = json_file["INR_trainer_config"]

    def fit_inr(self, dataset, index, model_cls, seed, save=False, save_name=None):
        img, _ = dataset[index]
        height, width = img.shape[1], img.shape[2]

        coords = [[i, j] for i in range(height) for j in range(width)]
        intensities = [img[:, i, j].item() for i in range(height) for j in range(width)]
        coords = torch.tensor(coords, dtype=torch.float32)
        intensities = torch.tensor(intensities, dtype=torch.float32)

        model = model_cls(seed=seed, INR_model_config=self.INR_model_config)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.INR_trainer_config["lr"])

        epochs = self.INR_trainer_config["epochs"]
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(coords)
            loss = criterion(outputs.squeeze(), intensities)
            loss.backward()
            optimizer.step()

            if self.debug and epoch % 50 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        if save:
            torch.save(model.state_dict(), save_name)

        return model.state_dict()

    def fit_inrs(self, dataset, indices, model_cls, seed):
        models = []
        for index in indices:
            final_model = self.fit_inr(dataset, index, model_cls, seed, save=True, save_name=f"./data/INR/sMLP_comparison/sMLP500_{index}")
            models.append(final_model)
        return models