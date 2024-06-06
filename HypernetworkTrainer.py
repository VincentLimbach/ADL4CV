import json
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


from architectures import HyperNetworkMLPGeneralExtended, HyperNetworkMLPConcat, sMLP
from data_access import *
from INRTrainer import INRTrainer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils import *

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
class PairedDataset(Dataset):
    def __init__(self, dataset, index_pairs):
        self.dataset = dataset
        self.index_pairs = index_pairs
        self._cache = {}

    def _get_item_from_cache(self, index):
        if index not in self._cache:
            img, label, flat_weights = self.dataset[index]
            label = torch.tensor(label, dtype=torch.int8)
            self._cache[index] = (img.squeeze(0), label.squeeze(0), flat_weights)
        return self._cache[index]

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        index_1, index_2 = self.index_pairs[idx]
        img_1, label_1, flat_weights_1 = self._get_item_from_cache(index_1)
        img_2, label_2, flat_weights_2 = self._get_item_from_cache(index_2)
        return img_1, label_1, flat_weights_1, img_2, label_2, flat_weights_2

class HyperNetworkTrainer:
    def __init__(self, hypernetwork, base_model_cls, inr_trainer, save_path, load=False, override=False):
        with open("config.json") as file:
            self.INR_model_config = json.load(file)["INR_model_config"]
        
        self.inr_trainer = inr_trainer
        self.hypernetwork = hypernetwork.to(device)  # Move hypernetwork to GPU
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


    def process_batch(self, batch, criterion, epoch, debug=False, final=False):
        start_time = time.time()
        img_1_batch, label_1_batch, flat_weights_1_batch, img_2_batch, label_2_batch, flat_weights_2_batch = [b.to(device) for b in batch]  # Move batch data to GPU
        concatenated_weights = torch.cat((flat_weights_1_batch, flat_weights_2_batch), dim=1)
        #concatenated_weights += torch.randn_like(concatenated_weights) * 0.05
        batch_size = img_1_batch.shape[0]
        new_model = self.base_model_cls(seed=42, INR_model_config=self.INR_model_config, device=device).to(device)

        offsets = torch.randint(1, 16, (batch_size, 4), device=device) * 2

        #print(f"Preparation took: {time.time() - start_time:.8f} seconds")
        start_time = time.time()

        predicted_weights = self.hypernetwork(concatenated_weights, offsets, label_1_batch, label_2_batch)
        #print(f"Predicting took: {time.time() - start_time:.8f} seconds")
        start_time = time.time()

        weights, biases = unflatten_weights(predicted_weights, new_model)
        img_concat = generate_merged_image(img_1_batch, img_2_batch, offsets[:, 0], offsets[:, 1], offsets[:, 2], offsets[:, 3])
        height, width = img_concat.shape[1], img_concat.shape[2]
        coords = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width)), dim=-1).reshape(-1, 2).float().to(device)
        intensities = img_concat.view(-1, batch_size).to(device)
        
        predictions = new_model(coords, weights=weights, biases=biases)
        
        loss = criterion(predictions.flatten(), intensities.flatten())
        #print(f"Calculating the loss took: {time.time() - start_time:.8f} seconds")
        if final:
            actual_images = predictions.view(batch_size, height, width)
            expected_images = img_concat
    
            results = []
            for i in range(batch_size):
                result = {
                    "actual": actual_images[i, :, :].cpu().detach().numpy(),
                    "expected": expected_images[i, :, :].cpu().detach().numpy()
                }
                results.append(result)
        else:
            results = []
        #print(f"Calculations per batch take: {time.time() - start_time:.8f} seconds")
        #print(f"Calculations per item take: {(time.time() - start_time)/batch_size:.8f} seconds")
        return loss.mean(), results

    def train_one_epoch(self, dataloader, criterion, optimizer, epoch, debug=False):
        self.hypernetwork.train()
        total_loss = 0.0
        counter = 0
        for batch in dataloader:
            counter+=1
            optimizer.zero_grad()
            start_time=time.time()
            batch_loss, _ = self.process_batch(batch, criterion, epoch, debug=debug)
            #print(f"Calculations for batch took: {time.time() - start_time:.8f} seconds")
            total_loss += batch_loss
            batch_loss.backward()
            #print(f"Backwards took: {time.time() - start_time:.8f} seconds")
            optimizer.step()
            #print(f"Optimizer step took: {time.time() - start_time:.8f} seconds")
        #print(counter)
        total_loss /= len(dataloader)
        return total_loss

    def validate(self, dataloader, criterion, epoch, debug, final):
        self.hypernetwork.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                batch_loss, results = self.process_batch(batch, criterion, epoch, debug=debug, final=final)
                total_loss += batch_loss

        total_loss /= len(dataloader)
        return total_loss, results

    def train_hypernetwork(self, dataset_name, train_pairs, val_pairs, on_the_fly=False, debug=False, batch_size=128, path_dic=None, path_model=None):
        dataset = ImageINRDataset(dataset_name, self.base_model_cls, self.inr_trainer, "data/INR/sMLP/", on_the_fly)
        if path_dic is not None:
            dataset = ImageINRDataset(dataset_name, self.base_model_cls, self.inr_trainer, path_dic, on_the_fly, path_model)

        train_dataset = PairedDataset(dataset, train_pairs)
        val_dataset = PairedDataset(dataset, val_pairs)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        if self.load:
            optimizer = torch.optim.Adam(self.hypernetwork.parameters(), lr=0.00005)
            epochs = 20

            for epoch in range(epochs):
                start_time = time.time()
                if (epoch + 1) % 1 == 0 or epoch == 0:
                    epoch_debug = debug
                else:
                    epoch_debug = False

                if epoch_debug:
                    print(f"[DEBUG] Starting epoch {epoch + 1}")


                train_loss = self.train_one_epoch(train_loader, criterion, optimizer, epoch, debug=epoch_debug)
                final = (epoch == epochs-1)
                val_loss, results = self.validate(val_loader, criterion, epoch, debug=epoch_debug, final=final)
                self.writer.add_scalars('Loss', {'train': train_loss.item(), 'val': val_loss.item()}, epoch)

                if epoch_debug:
                    print(f"Calculations for epoch {epoch} took: {time.time() - start_time:.8f} seconds")

                if (epoch + 1) % 1 == 0 or epoch == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss.item()}, Val: {val_loss.item()}")
                
            torch.save(self.hypernetwork.state_dict(), self.save_path)

        print(f"Final_validation: {val_loss}")
        return results

def main():
    with open("./config.json", "r") as json_file:
        json_file = json.load(json_file)
        INR_model_config = json_file["INR_model_config"]
        INR_dataset_config = json_file["INR_dataset_config"]
    
    inr_trainer = INRTrainer()
    hypernetwork = HyperNetworkMLPGeneralExtended().to(device) 

    hypernetwork_trainer = HyperNetworkTrainer(hypernetwork, sMLP, inr_trainer, save_path='models/hypernetwork_big_8192_both_arbitrary_positive_extended.pth', load=True)
    train_pairs, val_pairs = generate_pairs(8192, 256, 16, seed=42)
    print(len(train_pairs))
    print(len(val_pairs))
    #train_pairs = [(1,1)]
    #train_pairs = [(0,0)]
    val_pairs= [(i, i+1) for i in range(8192,8448,2)]
    print(len(val_pairs))
    results = hypernetwork_trainer.train_hypernetwork("MNIST", train_pairs=train_pairs, val_pairs=val_pairs, on_the_fly=True, debug=True)

    for index, ip in enumerate(results):
        expected_image = ip["expected"]
        actual_image = ip["actual"]

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(expected_image, cmap='gray')
        ax[0].set_title('Original Concatenated Image')
        ax[0].axis('off')

        ax[1].imshow(actual_image, cmap='gray')
        ax[1].set_title('Predicted Image')
        ax[1].axis('off')
        
        plt.savefig(f"evaluation/strong_validation_{index}.png")
        plt.close(fig)

if __name__ == "__main__":
    main()
