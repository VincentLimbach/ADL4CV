import json
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from ADL4CV.architectures import HyperNetwork3D, sMLP
from ADL4CV.data import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils import *
import mcubes
import trimesh

device = 'cpu'
with open("config.json") as file:
    new_model = sMLP(seed=42, INR_model_config=json.load(file)["INR_model_config_3D"], device=device)
print(device)

class PairedDataset3D(Dataset):
    def __init__(self, dataset, index_pairs):
        self.dataset = dataset
        self.index_pairs = index_pairs
        self._cache = {}

    def _get_item_from_cache(self, index):
        if index not in self._cache:
            obj, label, flat_weights = self.dataset[index]
            self._cache[index] = (obj.squeeze(0), label, flat_weights)
        return self._cache[index]

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        index_1, index_2 = self.index_pairs[idx]
        obj_1, label_1, flat_weights_1 = self._get_item_from_cache(index_1)
        obj_2, label_2, flat_weights_2 = self._get_item_from_cache(index_2)
        return obj_1, label_1, flat_weights_1, obj_2, label_2, flat_weights_2


class HyperNetworkTrainer:
    def __init__(self, hypernetwork, base_model_cls, inr_trainer, save_path, load=False, override=False):
        with open("config.json") as file:
            self.INR_model_config = json.load(file)["INR_model_config_3D"]
        
        self.inr_trainer = inr_trainer
        self.hypernetwork = hypernetwork.to(device) 
        self.base_model_cls = base_model_cls
        self.save_path = save_path
        self.load = load
        self.writer = SummaryWriter(f'ADL4CV/logs/3D/hypernetwork/')

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

    def process_batch(self, batch, criterion, debug=False, final=False):
        obj_1_batch, _, flat_weights_1_batch, obj_2_batch, _, flat_weights_2_batch = [b for b in batch]
        concatenated_weights = torch.cat((flat_weights_1_batch, flat_weights_2_batch), dim=1)
        batch_size = obj_1_batch.shape[0]
        offsets = torch.zeros((batch_size, 6))
        offsets[:, 3] = 1
        predicted_weights = self.hypernetwork(concatenated_weights) 

        weights, biases = unflatten_weights(predicted_weights, new_model)

        coords, sdfs = generate_merged_object(obj_1_batch, obj_2_batch, 0, -.5, 0, 0, .5, 0, device)

        coords = dict2cuda(coords, device)
        sdfs = dict2cuda(sdfs, device)

        predictions = new_model(coords, weights=weights, biases=biases)
        
        loss = criterion(predictions, sdfs)

        if final:    
            results = predictions
        else:
            results = []

        return loss.mean(), results


    def train_one_epoch(self, dataloader, criterion, optimizer, debug=False):
        self.hypernetwork.train()
        total_loss = 0.0
        counter = 0
        for batch in dataloader:
            counter+=1
            optimizer.zero_grad(set_to_none=True)
            batch_loss, _ = self.process_batch(batch, criterion, debug=debug)
            total_loss += batch_loss
            batch_loss.backward()
            optimizer.step()
        total_loss /= len(dataloader)
        return total_loss

    def validate(self, dataloader, criterion, debug, final):
        self.hypernetwork.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                batch_loss, results = self.process_batch(batch, criterion, debug=debug, final=final)
                total_loss += batch_loss

        total_loss /= len(dataloader)
        return total_loss, results


    def train_hypernetwork(self, train_pairs, val_pairs, on_the_fly=False, debug=False, batch_size=512, path_dic=None, path_model=None):
        dataset = ObjectINRDataset(sMLP, self.inr_trainer, "ADL4CV/data/model_data/shrec_16", on_the_fly=False, device=device)

        train_dataset = PairedDataset3D(dataset, train_pairs)
        val_dataset = PairedDataset3D(dataset, val_pairs)

        for idx, i in enumerate(train_dataset):
            obj_1, _, _, obj_2, _, _ = i
            if obj_1.shape[0] == 250:
                print(idx, train_pairs[idx], 1)
            if obj_2.shape[0] == 250:
                print(idx, train_pairs[idx], 2)


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()

        def lr_lambda(epoch):
            if epoch < 1000:
                return 1.0
            elif epoch < 2000:
                return 1/6
            else:
                return 1/12


        if not self.load:
            optimizer = torch.optim.Adam(self.hypernetwork.parameters(), lr=0.0005)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

            epochs = 5000

            for epoch in range(epochs):
                start_time = time.time()
                if (epoch + 1) % 1 == 0 or epoch == 0:
                    epoch_debug = debug
                else:
                    epoch_debug = False

                if epoch_debug:
                    print(f"[DEBUG] Starting epoch {epoch + 1}")


                train_loss = self.train_one_epoch(train_loader, criterion, optimizer, debug=epoch_debug)
                final = (epoch == epochs-1)
                val_loss, _ = self.validate(val_loader, criterion, debug=epoch_debug, final=final)
                self.writer.add_scalars('Loss', {'train': train_loss.item(), 'val': val_loss.item()}, epoch)

                if epoch_debug:
                    print(f"Calculations for epoch {epoch} took: {time.time() - start_time:.8f} seconds")

                if (epoch + 1) % 1 == 0 or epoch == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss.item()}, Val: {val_loss.item()}")
                scheduler.step()

            torch.save(self.hypernetwork.state_dict(), self.save_path)

        return


def main():
    hypernetwork = HyperNetwork3D()
    
    inr_trainer = INRTrainer3D(debug=True)
    hypernetwork_trainer = HyperNetworkTrainer(hypernetwork, sMLP, inr_trainer, save_path='ADL4CV/models/3D/hypernetwork_large.pth', override=True)
    train_pairs, val_pairs = generate_pairs(256, 8, 16)

    dataset = ObjectINRDataset(sMLP, inr_trainer, "ADL4CV/data/model_data/shrec_16", on_the_fly=False, device=device)
    
    outliers = []
    for i in range(256):
        if dataset[i][0].shape[0] != 252:
            print(i, dataset[i][0].shape[0])
            outliers.append(i)

    print("Non conforming objects: ", outliers)
    
    
    #------------------------------- Hypernetwork funktionier auf diesem Training set, da keine objekte mit 250 vertices enthalten -----------
    armadillo_train_idx = [54, 61, 201, 226, 260 ] #, 279, 341, 344, 370,
#                         403, 457, 489] #, 522, 539, 558, 597]
    armadillo_pairs = [[x, y] for x in armadillo_train_idx for y in armadillo_train_idx]
    random.seed(43)
    random.shuffle(armadillo_pairs)
    train_pairs = armadillo_pairs[:20]
    val_pairs = armadillo_pairs[20:]

    #-----------------------------------------------------------------------------------------


    #train_pairs = [(61,61) for i in range(16)]
    #val_pairs = [(61,61) for i in range(2)]

    hypernetwork_trainer.train_hypernetwork(train_pairs, val_pairs, batch_size=2)







    
    hypernetwork_trainer.train_hypernetwork(train_pairs, val_pairs, batch_size=32)
    
    hypernetwork.load_state_dict(torch.load("ADL4CV/models/3D/hypernetwork_large.pth", map_location=torch.device(device)))

    with open("config.json") as json_file:
                json_file = json.load(json_file)
                INR_model_config = json_file["INR_model_config_3D"]

    model = sMLP(seed=42, INR_model_config=INR_model_config)

    idx_1 , idx_2 = train_pairs[5]
    print(idx_1, idx_2)

    model.load_state_dict(torch.load(f"ADL4CV/data/model_data/shrec_16/T{idx_1}.pth", map_location=torch.device(device)))
    params_1 = flatten_model_weights(model)
    model.load_state_dict(torch.load("ADL4CV/data/model_data/shrec_16/T{idx_2}.pth", map_location=torch.device(device)))
    params_2 = flatten_model_weights(model)
    params_cat = torch.cat((params_2, params_1)).to(device)

    predicted_weights = hypernetwork(params_cat).unsqueeze(0)

    weights, biases = unflatten_weights(predicted_weights, model)
    
    step = .01
    scale_1 = 2
    scale_2 = 1
    render_and_store_models(step, scale_1, scale_2, scale_2, model_path=None, weights=weights, biases=biases)



if __name__ == "__main__":
    main()

