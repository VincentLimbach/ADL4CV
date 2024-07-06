import json
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from ADL4CV.architectures import HyperNetworkTrueRes, sMLP, SharpNet
from ADL4CV.data import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils import *
from datetime import datetime

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open("config.json") as file:
    new_model = sMLP(seed=42, INR_model_config=json.load(file)["INR_model_config_2D"], device=device)
print(device)

class PairedDataset(Dataset):
    def __init__(self, dataset, index_pairs):
        self.dataset = dataset
        self.index_pairs = index_pairs
        self._cache = {}

    def _get_item_from_cache(self, index):
        if index not in self._cache:
            img, label, flat_weights = self.dataset[index]
            self._cache[index] = (img.squeeze(0), label.squeeze(0), flat_weights)
        return self._cache[index]

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        index_1, index_2 = self.index_pairs[idx]
        img_1, label_1, flat_weights_1 = self._get_item_from_cache(index_1)
        img_2, label_2, flat_weights_2 = self._get_item_from_cache(index_2)
        return img_1, label_1, flat_weights_1, img_2, label_2, flat_weights_2

class SharpnetTrainer:
    def __init__(self, sharpnet, hypernetwork, base_model_cls, inr_trainer, save_path, hypernet_path, load=False, override=False):
        with open("config.json") as file:
            self.INR_model_config = json.load(file)["INR_model_config_2D"]
        
        self.sharpnet = sharpnet
        self.inr_trainer = inr_trainer
        self.hypernetwork = hypernetwork.to(device)  # Move hypernetwork to GPU
        self.base_model_cls = base_model_cls
        self.save_path = save_path
        self.hypernet_path = hypernet_path
        self.load = load
        self.writer = SummaryWriter(f'ADL4CV/logs/2D/hypernetwork/')

        self.image_cache = {}
        self.dataset_cache = {}
        if load:
            if os.path.exists(self.hypernet_path):
                self.hypernetwork.load_state_dict(torch.load(self.hypernet_path, map_location=torch.device('cpu')))
                print(f"Model loaded from {self.hypernet_path}")
            else:
                raise FileNotFoundError(f"No model found at {self.hypernet_path} to load.")
        elif not override and os.path.exists(self.hypernet_path):
            raise FileExistsError(f"Model already exists at {self.hypernet_path}. Use override=True to overwrite it.")


    def process_batch(self, batch, criterion, epoch, debug=False, final=False):
        img_1_batch, label_1_batch, flat_weights_1_batch, img_2_batch, label_2_batch, flat_weights_2_batch = [b for b in batch]   # Move batch data to GPU
        concatenated_weights = torch.cat((flat_weights_1_batch, flat_weights_2_batch), dim=1)
        batch_size = img_1_batch.shape[0]
        offsets = torch.randint(1, 16, (batch_size, 4), device=device) * 2
        predicted_weights = self.hypernetwork(concatenated_weights, offsets, label_1_batch, label_2_batch)

        weights, biases = unflatten_weights(predicted_weights, new_model)

        img_concat = generate_merged_image(img_1_batch, img_2_batch, offsets[:, 0], offsets[:, 1], offsets[:, 2], offsets[:, 3], device)
        height, width = img_concat.shape[1], img_concat.shape[2]
        coords = torch.cartesian_prod(torch.arange(height, device=device), torch.arange(width, device=device)).float()

        intensities = img_concat.view(-1, batch_size)
        predictions_hypernet = new_model(coords, weights=weights, biases=biases)
        
        print(predictions_hypernet.shape)
        predictions_sharpnet = self.sharpnet(predictions_hypernet)
        print(predictions_sharpnet.shape)
        loss = criterion(predictions_sharpnet.flatten(), intensities.flatten())

        if final:
            actual_images = predictions_sharpnet.view(batch_size, height, width)
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
        return loss.mean(), results

    def train_one_epoch(self, dataloader, criterion, optimizer, epoch, debug=False):
        self.sharpnet.train()
        total_loss = 0.0
        counter = 0
        for batch in dataloader:
            counter+=1
            optimizer.zero_grad(set_to_none=True)
            batch_loss, _ = self.process_batch(batch, criterion, epoch, debug=debug)
            total_loss += batch_loss
            batch_loss.backward()
            optimizer.step()
        total_loss /= len(dataloader)
        return total_loss

    def validate(self, dataloader, criterion, epoch, debug, final):
        self.sharpnet.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                batch_loss, results = self.process_batch(batch, criterion, epoch, debug=debug, final=final)
                total_loss += batch_loss

        total_loss /= len(dataloader)
        return total_loss, results

    def train_sharpnet(self, dataset_name, train_pairs, weak_val_pairs, strong_val_pairs, on_the_fly=False, debug=False, batch_size=512, path_dic=None, path_model=None):
        dataset = ImageINRDataset(dataset_name, self.base_model_cls, self.inr_trainer, "ADL4CV/data/model_data/MNIST", on_the_fly, device=device)
        if path_dic is not None:
            dataset = ImageINRDataset(dataset_name, self.base_model_cls, self.inr_trainer, path_dic, on_the_fly, path_model, device=device)

        train_dataset = PairedDataset(dataset, train_pairs)
        weak_val_dataset = PairedDataset(dataset, weak_val_pairs)
        strong_val_dataset = PairedDataset(dataset, strong_val_pairs)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        weak_val_loader = DataLoader(weak_val_dataset, batch_size=batch_size, shuffle=False)
        strong_val_loader = DataLoader(strong_val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.L1Loss()
        
        def lr_lambda(epoch):
            return 1 - (epoch/100)*(19/20)
            
        if not self.load:
            optimizer = optim.Adam(self.sharpnet.parameters(), lr=0.005)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

            epochs = 100

            for epoch in range(epochs):
                start_time = time.time()

                train_loss = self.train_one_epoch(train_loader, criterion, optimizer, epoch, debug=debug)
                final = (epoch == epochs-1)
                weak_val_loss, weak_val_results = self.validate(weak_val_loader, criterion, epoch, debug=debug, final=final)
                strong_val_loss, strong_val_results = self.validate(strong_val_loader, criterion, epoch, debug=debug, final=final)
                self.writer.add_scalars(datetime.now().strftime("%Y%m%d-%H%M%S"), {'Train': train_loss.item(), 'Weak_Val': weak_val_loss.item(), 'Strong_Val': strong_val_loss.item()}, epoch)

                if epoch == 0 or epoch == 1 or epoch%5==0:
                    print(f"Calculations for epoch {epoch} took: {time.time() - start_time:.8f} seconds")

                if (epoch + 1) % 1 == 0 or epoch == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Train: {train_loss.item()}, Weak_Val: {weak_val_loss.item()}, Strong_Val: {strong_val_loss.item()}")
                scheduler.step()

            torch.save(self.sharpnet.state_dict(), self.save_path)

        print(f"Final\nWeak validation: {weak_val_loss}\nStrong validation: {strong_val_loss}")
        return weak_val_results, strong_val_results

def save_figures(path, figures):
    for index, ip in enumerate(figures):
        expected_image = ip["expected"]
        actual_image = ip["actual"]

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(expected_image, cmap='gray')
        ax[0].set_title('Original Concatenated Image')
        ax[0].axis('off')

        ax[1].imshow(actual_image, cmap='gray')
        ax[1].set_title('Predicted Image')
        ax[1].axis('off')
        
        plt.savefig(f"{path}{index}.png")
        plt.close(fig)

def main():
    with open("./config.json", "r") as json_file:
        json_file = json.load(json_file)
    
    inr_trainer = INRTrainer2D()
    hypernetwork = HyperNetworkTrueRes().to(device) 
    sharpnet = SharpNet().to(device)

    sharpnet_trainer = SharpnetTrainer(sharpnet, hypernetwork, sMLP, inr_trainer, save_path='ADL4CV/models/2D/sharpnet.pth', hypernet_path="ADL4CV/models/2D/hypernetwork_true_res.pth", load=True, override=True)
    train_pairs, weak_val_pairs = generate_pairs(8192, 256, 32, seed=42)
    strong_val_pairs, _ = generate_pairs(256, 0, 1, seed=42)
    strong_val_pairs= [(i+8192, j+8192) for i, j in strong_val_pairs]
    print(len(train_pairs))
    print(len(weak_val_pairs))
    print(len(strong_val_pairs))

    weak_val_results, strong_val_results = sharpnet.train_sharpnet("MNIST", train_pairs=train_pairs, weak_val_pairs=weak_val_pairs, strong_val_pairs=strong_val_pairs, on_the_fly=True, debug=True)

    save_figures("ADL4CV/evaluation/sharpnet/weak_validation/", weak_val_results)
    save_figures("ADL4CV/evaluation/sharpnet/strong_validation/", strong_val_results)

if __name__ == "__main__":
    main()

