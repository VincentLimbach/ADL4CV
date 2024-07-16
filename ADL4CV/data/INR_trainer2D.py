import json
import time

import torch
import torch.nn as nn
import torch.optim as optim


class INRTrainer2D:
    def __init__(self, debug=True):
        self.debug = debug
        with open("config.json") as json_file:
            json_file = json.load(json_file)
            self.INR_model_config = json_file["INR_model_config_2D"]
            self.INR_trainer_config = json_file["INR_trainer_config_2D"]

    def fit_inr(self, dataset, index, model_cls, seed, save=False, save_name=None):
        img, _ = dataset[index]
        height, width = img.shape[1], img.shape[2]

        coords = [[i, j] for i in range(height) for j in range(width)]
        intensities = [img[:, i, j].item() for i in range(height) for j in range(width)]
        coords = torch.tensor(coords, dtype=torch.float32)
        intensities = torch.tensor(intensities, dtype=torch.float32)

        model = model_cls(seed=seed, INR_model_config=self.INR_model_config)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.INR_trainer_config["lr"],
            weight_decay=self.INR_trainer_config.get("weight_decay", 0),
        )

        epochs = self.INR_trainer_config["epochs"]
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(coords)
            loss = criterion(outputs.squeeze(), intensities)
            loss.backward()
            optimizer.step()

            if self.debug and epoch % 50 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        if save:
            torch.save(model.state_dict(), save_name)

        return model.state_dict()

    def fit_inrs(self, dataset, indices, model_cls, seed, save_path):
        models = []
        for index in indices:
            final_model = self.fit_inr(
                dataset,
                index,
                model_cls,
                seed,
                save=True,
                save_name=save_path + str(index) + ".pth",
            )
            models.append(final_model)

        return models
