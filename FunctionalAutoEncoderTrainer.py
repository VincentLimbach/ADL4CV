from utils import flatten_model_weights, unflatten_weights, generate_coordinates
from architectures import sMLP, Autoencoder
from ImageINRDataset import ImageINRDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from INRTrainer import INRTrainer
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

class FunctionalAutoencoderTrainer:
    def __init__(self, autoencoder, dataset, base_model_cls=sMLP, arg_dict=None, batch_size=32, lr=0.001, epochs=50, save_dir="models/autoencoder/"):
        self.autoencoder = autoencoder
        self.dataset = dataset
        self.base_model_cls = base_model_cls
        self.arg_dict = arg_dict if arg_dict is not None else {}
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.save_dir=save_dir

        train_size = int(0.92 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
        self.writer = SummaryWriter(log_dir="runs/autoencoder_experiment")

    def collate_fn(self, batch):
        images, flat_weights_list = zip(*[self.dataset.__getitem__(index) for index in range(len(batch))])
        images = torch.stack(images)
        flat_weights_list = torch.stack(flat_weights_list)
        return images, flat_weights_list

    def process_batch(self, batch, training=True):
        images, original_weights = batch
        batch_size = len(images)

        if training:
            self.optimizer.zero_grad()

        predicted_weights = self.autoencoder(original_weights)
        batch_loss = 0.0

        for img, original_weight, predicted_weight in zip(images, original_weights, predicted_weights):
            coordinates = generate_coordinates(img)

            original_model = self.base_model_cls(seed=42, arg_dict=self.arg_dict)
            original_model.load_state_dict(unflatten_weights(original_weight, original_model))
            original_preds = original_model(coordinates)

            reconstructed_model = self.base_model_cls(seed=42, arg_dict=self.arg_dict)
            reconstructed_preds = reconstructed_model(coordinates, external_parameters=unflatten_weights(predicted_weight, reconstructed_model))

            loss = self.criterion(reconstructed_preds, original_preds)
            batch_loss += loss

        batch_loss /= batch_size

        if training:
            batch_loss.backward()
            self.optimizer.step()

        return batch_loss.item()

    def train(self):
        for epoch in range(self.epochs):
            total_train_loss = 0.0
            self.autoencoder.train()

            for batch in self.train_loader:
                batch_loss = self.process_batch(batch, training=True)
                total_train_loss += batch_loss

            total_train_loss /= len(self.train_loader)

            total_val_loss = self.validate()

            self.writer.add_scalars('Loss', {'train': total_train_loss, 'val': total_val_loss}, epoch)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_train_loss:.8f}, Val Loss: {total_val_loss:.8f}")

            if epoch % 10 == 0:
                images, original_weights = next(iter(self.train_loader))
                predicted_weights = self.autoencoder(original_weights)
                self.log_predictions(images, original_weights, predicted_weights, epoch)

        torch.save(self.autoencoder.state_dict(), self.save_dir + "autoencoder.pth")
        self.writer.close()

    def validate(self):
        self.autoencoder.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                batch_loss = self.process_batch(batch, training=False)
                total_val_loss += batch_loss

        total_val_loss /= len(self.val_loader)
        return total_val_loss

    def log_predictions(self, images, original_weights, predicted_weights, epoch):
        fig, axs = plt.subplots(10, 2, figsize=(8, 8))
        for i in range(10):
            coordinates = generate_coordinates(images[i])

            original_model = self.base_model_cls(seed=42, arg_dict=self.arg_dict)
            original_model.load_state_dict(unflatten_weights(original_weights[i], original_model))
            original_preds = original_model(coordinates).detach().cpu().numpy()

            reconstructed_model = self.base_model_cls(seed=42, arg_dict=self.arg_dict)
            reconstructed_preds = reconstructed_model(coordinates, external_parameters=unflatten_weights(predicted_weights[i], reconstructed_model)).detach().cpu().numpy()

            axs[i, 0].imshow(original_preds.reshape(images[i].shape[1], images[i].shape[2]), cmap='gray')
            axs[i, 0].set_title("Original Prediction")
            axs[i, 1].imshow(reconstructed_preds.reshape(images[i].shape[1], images[i].shape[2]), cmap='gray')
            axs[i, 1].set_title("Reconstructed Prediction")

        plt.tight_layout()
        self.writer.add_figure(f'Predictions_epoch_{epoch}', fig)
        print(f"Logged predictions for epoch {epoch}")

arg_dict = {
        'input_feature_dim': 2,
        'output_feature_dim': 1,
        'hidden_features': 64,
        'layers': 2,
        'positional': True,
        'd_model': 16
    }

trainer = INRTrainer(subdirectory='/sMLP')
subset_indices = list(range(500))
dataset = Subset(ImageINRDataset("MNIST", sMLP, arg_dict, trainer, on_the_fly=True), subset_indices)
autoencoder = Autoencoder()
trainer = FunctionalAutoencoderTrainer(autoencoder, dataset, sMLP, arg_dict)
trainer.train()