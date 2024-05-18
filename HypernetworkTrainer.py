import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from architectures import sMLP, HyperNetworkMLP
from INRTrainer import INRTrainer
from ImageINRDataset import ImageINRDataset
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def process_predictions(predictions):
    return predictions
    """height, width = predictions.shape
    
    processed_predictions = np.copy(predictions)
    
    for i in range(height):
        for j in range(width):
            current_value = predictions[i, j]
            has_valid_neighbor = False
            
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    
                    ni, nj = i + di, j + dj
                    
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_value = predictions[ni, nj]
                        
                        if abs(current_value - neighbor_value) <= 0.2 or current_value>=0.3:
                            has_valid_neighbor = True
                            break
                if has_valid_neighbor:
                    break
            
            if not has_valid_neighbor:
                processed_predictions[i, j] = 0
    
    return processed_predictions"""


class HyperNetworkTrainer:
    def __init__(self, hypernetwork, base_model_cls, arg_dict, trainer, save_path, load=False, override=False):
        self.hypernetwork = hypernetwork
        self.base_model_cls = base_model_cls
        self.arg_dict = arg_dict
        self.trainer = trainer
        self.save_path = save_path
        self.load = load
        self.writer = SummaryWriter('runs/hypernetwork_experiment')
        if load:
            if os.path.exists(self.save_path):
                self.hypernetwork.load_state_dict(torch.load(self.save_path, map_location=torch.device('cpu')))
                print(f"Model loaded from {self.save_path}")
            else:
                raise FileNotFoundError(f"No model found at {self.save_path} to load.")
        elif not override and os.path.exists(self.save_path):
            raise FileExistsError(f"Model already exists at {self.save_path}. Use override=True to overwrite it.")


    def flatten_model_weights(self, model):
        flat_weights = []
        for layer in model.layers:
            flat_weights.append(layer.weight.data.view(-1))
            flat_weights.append(layer.bias.data.view(-1))
        return torch.cat(flat_weights)

    def unflatten_weights(self, flat_weights, model):
        external_parameters = {}
        offset = 0
        for i, layer in enumerate(model.layers):
            weight_shape = layer.weight.shape
            bias_shape = layer.bias.shape
            weight_numel = layer.weight.numel()
            bias_numel = layer.bias.numel()

            external_parameters[f"layers.{i}.weight"] = flat_weights[offset:offset + weight_numel].view(weight_shape)
            offset += weight_numel
            external_parameters[f"layers.{i}.bias"] = flat_weights[offset:offset + bias_numel].view(bias_shape)
            offset += bias_numel

        return external_parameters

    def process_batch(self, dataset, index_1, index_2, criterion):
        img_1, model_1 = dataset[index_1]
        img_2, model_2 = dataset[index_2]

        flat_weights_1 = self.flatten_model_weights(model_1)
        flat_weights_2 = self.flatten_model_weights(model_2)

        concatenated_weights = torch.cat((flat_weights_1, flat_weights_2))
        concatenated_weights += torch.randn_like(concatenated_weights)*0.05
        predicted_weights = self.hypernetwork(concatenated_weights)
        new_model = self.base_model_cls(seed=42, arg_dict=self.arg_dict)
        external_parameters = self.unflatten_weights(predicted_weights, new_model)


        img_concat = torch.cat((img_1, img_2), dim=2)

        height, width = img_concat.shape[1], img_concat.shape[2]
        coords = [[i, j] for i in range(height) for j in range(width)]
        coords = torch.tensor(coords, dtype=torch.float32)

        intensities = [img_concat[:, i, j].item() for i in range(height) for j in range(width)]
        intensities = torch.tensor(intensities, dtype=torch.float32)

        predictions = new_model(coords, external_parameters=external_parameters)
        loss = criterion(predictions.squeeze(), intensities)

        return loss, img_concat, predictions

    def train_one_epoch(self, dataset, index_pairs, criterion, optimizer):
        self.hypernetwork.train()
        total_loss = 0.0

        for index_1, index_2 in index_pairs:
            optimizer.zero_grad()
            loss, _, _ = self.process_batch(dataset, index_1, index_2, criterion)
            total_loss += loss
            loss.backward()
            optimizer.step()

        total_loss /= len(index_pairs)
        return total_loss

    def validate(self, dataset, index_pairs, criterion):
        self.hypernetwork.eval()
        total_loss = 0.0

        with torch.no_grad():
            for index_1, index_2 in index_pairs:
                loss, _, _ = self.process_batch(dataset, index_1, index_2, criterion)
                total_loss += loss

        total_loss /= len(index_pairs)
        return total_loss

    def train_hypernetwork(self, dataset_name, train_pairs, val_pairs, on_the_fly=False):
        dataset = ImageINRDataset(dataset_name, self.base_model_cls, self.arg_dict, self.trainer, on_the_fly)
        criterion = nn.MSELoss()
        if not self.load:
            optimizer = torch.optim.Adam(self.hypernetwork.parameters(), lr=0.0003, weight_decay=1e-6)
            epochs = 500

            for epoch in range(epochs):
                train_loss = self.train_one_epoch(dataset, train_pairs, criterion, optimizer)
                val_loss = self.validate(dataset, val_pairs, criterion)
                
                self.writer.add_scalar('Loss/train', train_loss.item(), epoch)
                self.writer.add_scalar('Loss/val', val_loss.item(), epoch)
                if (epoch + 1) % 50 == 0 or epoch == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss.item()}, Val: {val_loss.item()}")


            torch.save(self.hypernetwork.state_dict(), self.save_path)

        results = []
        for index_1, index_2 in val_pairs:
            _, img_concat, predictions = self.process_batch(dataset, index_1, index_2, criterion)
            predictions = predictions.detach().numpy().reshape((img_concat.shape[1], img_concat.shape[2]))
            
            predictions = process_predictions(predictions)
    
            predictions = np.clip(predictions, 0, 1)
            
            results.append((img_concat, predictions))

        return results

def main():
    arg_dict = {
        'input_feature_dim': 2,
        'output_feature_dim': 1,
        'hidden_features': 64,
        'layers': 2,
        'positional': True,
        'd_model': 16
    }

    trainer = INRTrainer(subdirectory='/sMLP')

    input_dim = 4354
    output_dim = 4354 // 2
    hypernetwork = HyperNetworkMLP(input_dim=input_dim, output_dim=output_dim)
    hypernetwork_trainer = HyperNetworkTrainer(hypernetwork, sMLP, arg_dict, trainer, save_path='hypernetwork.pth', load=True)

    all_pairs = [(i, j) for i in range(25) for j in range(25)]

    val_pairs = random.sample(all_pairs, 32)

    #train_pairs = [pair for pair in all_pairs if pair not in val_pairs]

    #val_pairs = [(26,24)]
    train_pairs = []
    results = hypernetwork_trainer.train_hypernetwork("MNIST", train_pairs=train_pairs, val_pairs=val_pairs, on_the_fly=True)

    for img_concat, predictions in results:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(img_concat.squeeze(), cmap='gray')
        ax[0].set_title('Original Concatenated Image')
        ax[0].axis('off')

        ax[1].imshow(predictions, cmap='gray')
        ax[1].set_title('Predicted Image')
        ax[1].axis('off')

        plt.show()

if __name__ == "__main__":
    main()