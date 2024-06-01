import json
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures import HyperNetworkMLPConcat, HyperNetworkIFE, sMLP
from data_access import *
from INRTrainer import INRTrainer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils import *

from PIL import Image
os.makedirs('presentation', exist_ok=True)

def save_image(tensor, path):
    array = tensor.numpy()
    array = (array * 255).astype(np.uint8)
    image = Image.fromarray(array, mode='L')
    image.save(path)

def save_prediction(predictions, path):
    array = (predictions * 255).astype(np.uint8)
    image = Image.fromarray(array, mode='L')
    image.save(path)

class PairedDataset(Dataset):
    def __init__(self, dataset, index_pairs):
        self.dataset = dataset
        self.index_pairs = index_pairs
        self._cache = {}

    def _get_item_from_cache(self, index):
        if index not in self._cache:
            img, flat_weights = self.dataset[index]
            self._cache[index] = (img, flat_weights)
        return self._cache[index]

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        index_1, index_2 = self.index_pairs[idx]
        img_1, flat_weights_1 = self._get_item_from_cache(index_1)
        img_2, flat_weights_2 = self._get_item_from_cache(index_2)
        return img_1, flat_weights_1, img_2, flat_weights_2

class HyperNetworkTrainer:
    def __init__(self, hypernetwork, base_model_cls, inr_trainer, save_path, load=False, override=False):
        with open("config.json") as file:
            self.INR_model_config = json.load(file)["INR_model_config"]
        
        self.inr_trainer = inr_trainer
        self.hypernetwork = hypernetwork
        self.base_model_cls = base_model_cls
        self.save_path = save_path
        self.load = load
        self.writer = SummaryWriter('runs/hypernetwork_experiment')

        self.image_cache = {}
        self.dataset_cache = {}
        if load:
            if os.path.exists(self.save_path):
                self.hypernetwork.load_state_dict(torch.load(self.save_path, map_location=torch.device('cpu')))
                print(f"Model loaded from {self.save_path}")
            else:
                raise FileNotFoundError(f"No model found at {self.save_path} to load.")
        elif not override and os.path.exists(self.save_path):
            raise FileExistsError(f"Model already exists at {self.save_path}. Use override=True to overwrite it.")

    def process_batch(self, batch, criterion, epoch, debug=False):
        img_1_batch, flat_weights_1_batch, img_2_batch, flat_weights_2_batch = batch
        concatenated_weights = torch.cat((flat_weights_1_batch, flat_weights_2_batch), dim=1)
        concatenated_weights += torch.randn_like(concatenated_weights) * 0.05

        predicted_weights = self.hypernetwork(concatenated_weights)

        losses = []
        for i in range(len(img_1_batch)):
            new_model = self.base_model_cls(seed=42, INR_model_config=self.INR_model_config)
            external_parameters = unflatten_weights(predicted_weights[i], new_model)

            img_concat = torch.cat((img_1_batch[i], img_2_batch[i]), dim=2)
            height, width = img_concat.shape[1], img_concat.shape[2]
            coords = [[x, y] for x in range(height) for y in range(width)]
            coords = torch.tensor(coords, dtype=torch.float32)
            intensities = [img_concat[:, x, y].item() for x in range(height) for y in range(width)]
            intensities = torch.tensor(intensities, dtype=torch.float32)

            predictions = new_model(coords, external_parameters=external_parameters)
            loss = criterion(predictions.squeeze(), intensities)
            losses.append(loss)

        batch_loss = torch.stack(losses).mean()
        return batch_loss

    def train_one_epoch(self, dataloader, criterion, optimizer, epoch, debug=False):
        self.hypernetwork.train()
        total_loss = 0.0

        for batch in dataloader:
            optimizer.zero_grad()
            batch_loss = self.process_batch(batch, criterion, epoch, debug=debug)
            total_loss += batch_loss
            batch_loss.backward()

            for name, param in self.hypernetwork.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, global_step=epoch)

            optimizer.step()

        total_loss /= len(dataloader)
        return total_loss

    def validate(self, dataloader, criterion, epoch, debug):
        self.hypernetwork.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                batch_loss = self.process_batch(batch, criterion, epoch, debug=debug)
                total_loss += batch_loss

        total_loss /= len(dataloader)
        return total_loss

    def train_hypernetwork(self, dataset_name, train_pairs, val_pairs, on_the_fly=False, debug=False, batch_size=32):
        dataset = ImageINRDataset(dataset_name, self.base_model_cls, self.inr_trainer, "data/INR/sMLP/", on_the_fly)
        train_dataset = PairedDataset(dataset, train_pairs)
        val_dataset = PairedDataset(dataset, val_pairs)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        if not self.load:
            optimizer = torch.optim.Adam(self.hypernetwork.parameters(), lr=0.0003, weight_decay=1e-6)
            epochs = 51

            for epoch in range(epochs):
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    epoch_debug = debug
                else:
                    epoch_debug = False

                if epoch_debug:
                    print(f"[DEBUG] Starting epoch {epoch + 1}")

                start_time = time.time()
                train_loss = self.train_one_epoch(train_loader, criterion, optimizer, epoch, debug=epoch_debug)
                val_loss = self.validate(val_loader, criterion, epoch, debug=epoch_debug)

                self.writer.add_scalars('Loss', {'train': train_loss.item(), 'val': val_loss.item()}, epoch)

                if epoch_debug:
                    print(f"[DEBUG] Epoch {epoch + 1} completed in {time.time() - start_time:.4f} seconds")
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss.item()}, Val: {val_loss.item()}")

            torch.save(self.hypernetwork.state_dict(), self.save_path)
        
        
        results = []

        for idx, (index_1, index_2) in enumerate(val_pairs):
            img_1, flat_weights_1 = dataset[index_1]
            img_2, flat_weights_2 = dataset[index_2]

            concatenated_weights = torch.cat((flat_weights_1, flat_weights_2))
            concatenated_weights += torch.randn_like(concatenated_weights) * 0.05

            predicted_weights = self.hypernetwork(concatenated_weights.unsqueeze(0)).squeeze(0)

            new_model = self.base_model_cls(seed=42, INR_model_config=self.INR_model_config)
            external_parameters = unflatten_weights(predicted_weights, new_model)

            img_concat = torch.cat((img_1, img_2), dim=2)
            height, width = img_concat.shape[1], img_concat.shape[2]
            coords = [[x, y] for x in range(height) for y in range(width)]
            coords = torch.tensor(coords, dtype=torch.float32)
            intensities = [img_concat[:, x, y].item() for x in range(height) for y in range(width)]
            intensities = torch.tensor(intensities, dtype=torch.float32)

            predictions = new_model(coords, external_parameters=external_parameters).detach().numpy().reshape((height, width))

            predictions = np.clip(predictions, 0, 1)
            results.append((img_concat, predictions))

            save_image(img_1.squeeze(0), f'presentation/org_{index_1}.png')
            save_image(img_2.squeeze(0), f'presentation/org_{index_2}.png')

            save_image(img_concat.squeeze(0), f'presentation/concat_{index_1}_{index_2}.png')

            save_prediction(predictions, f'presentation/strong_val_{index_1}_{index_2}.png')

        return results


def main():
    with open("./config.json", "r") as json_file:
        json_file = json.load(json_file)
        INR_model_config = json_file["INR_model_config"]
        INR_dataset_config = json_file["INR_dataset_config"]
    
    inr_trainer = INRTrainer()
    hypernetwork = HyperNetworkMLPConcat()

    hypernetwork_trainer = HyperNetworkTrainer(hypernetwork, sMLP, inr_trainer, save_path='models/hypernetwork_2000_3_pres.pth', load=True)

    train_pairs, val_pairs = generate_pairs(2000, 64, 8)

    results = hypernetwork_trainer.train_hypernetwork("MNIST", train_pairs=train_pairs, val_pairs=val_pairs, on_the_fly=True, debug=True)

    for img_concat, predictions in results:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(img_concat.squeeze(), cmap='gray')
        ax[0].set_title('Original Concatenated Image')
        ax[0].axis('off')

        ax[1].imshow(predictions, cmap='gray')
        ax[1].set_title('Predicted Image')
        ax[1].axis('off')
        plt.savefig("Test")
        plt.show()


if __name__ == "__main__":
    main()