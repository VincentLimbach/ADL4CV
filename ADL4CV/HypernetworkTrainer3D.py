import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from ADL4CV.architectures import HyperNetwork3D, sMLP
from ADL4CV.data import *
from ADL4CV.utils import *


device = "cpu"
with open("config.json") as file:
    new_model = sMLP(
        seed=42, INR_model_config=json.load(file)["INR_model_config_3D"], device=device
    )
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
    def __init__(
        self,
        hypernetwork,
        base_model_cls,
        inr_trainer,
        save_path,
        load=False,
        override=False,
    ):
        with open("config.json") as file:
            self.INR_model_config = json.load(file)["INR_model_config_3D"]

        self.inr_trainer = inr_trainer
        self.hypernetwork = hypernetwork.to(device)
        self.base_model_cls = base_model_cls
        self.save_path = save_path
        self.load = load
        self.writer = SummaryWriter(f"ADL4CV/logs/3D/hypernetwork/")

        self.image_cache = {}
        self.dataset_cache = {}
        if load:
            if os.path.exists(self.save_path):
                self.hypernetwork.load_state_dict(
                    torch.load(self.save_path, map_location=torch.device("cpu"))
                )
                print(f"Model loaded from {self.save_path}")
            else:
                raise FileNotFoundError(f"No model found at {self.save_path} to load.")
        elif not override and os.path.exists(self.save_path):
            raise FileExistsError(
                f"Model already exists at {self.save_path}. Use override=True to overwrite it."
            )

    def process_batch(self, batch, criterion, debug=False, final=False):
        obj_1_batch, _, flat_weights_1_batch, obj_2_batch, _, flat_weights_2_batch = [
            b for b in batch
        ]
        concatenated_weights = torch.cat(
            (flat_weights_1_batch, flat_weights_2_batch), dim=1
        )
        batch_size = obj_1_batch.shape[0]
        offsets = torch.zeros((batch_size, 6))
        offsets[:, 3] = 1
        predicted_weights = self.hypernetwork(concatenated_weights)

        weights, biases = unflatten_weights(predicted_weights, new_model)

        coords, sdfs = generate_merged_object(
            obj_1_batch, obj_2_batch, 0, -0.5, 0, 0, 0.5, 0, device
        )

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
            counter += 1
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
                batch_loss, results = self.process_batch(
                    batch, criterion, debug=debug, final=final
                )
                total_loss += batch_loss

        total_loss /= len(dataloader)
        return total_loss, results

    def train_hypernetwork(
        self,
        train_pairs,
        val_pairs,
        on_the_fly=False,
        debug=False,
        batch_size=512,
        path_dic=None,
        path_model=None,
    ):
        dataset = ObjectINRDataset(
            sMLP,
            self.inr_trainer,
            "ADL4CV/data/model_data/shrec_16",
            on_the_fly=False,
            device=device,
        )

        train_dataset = PairedDataset3D(dataset, train_pairs)
        val_dataset = PairedDataset3D(dataset, val_pairs)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()

        epochs = 5000

        def lr_lambda(epoch):
            return 1.0 - epoch / epochs * 11 / 12

        if not self.load:
            optimizer = torch.optim.Adam(
                self.hypernetwork.parameters(), lr=0.00005, weight_decay=0.002
            )
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

            epochs = 2000

            for epoch in range(epochs):
                start_time = time.time()
                if (epoch + 1) % 1 == 0 or epoch == 0:
                    epoch_debug = debug
                else:
                    epoch_debug = False

                if epoch_debug:
                    print(f"[DEBUG] Starting epoch {epoch + 1}")

                train_loss = self.train_one_epoch(
                    train_loader, criterion, optimizer, debug=epoch_debug
                )
                final = epoch == epochs - 1
                val_loss, _ = self.validate(
                    val_loader, criterion, debug=epoch_debug, final=final
                )
                self.writer.add_scalars(
                    "Loss", {"train": train_loss.item(), "val": val_loss.item()}, epoch
                )

                if epoch_debug:
                    print(
                        f"Calculations for epoch {epoch} took: {time.time() - start_time:.8f} seconds"
                    )

                if (epoch + 1) % 1 == 0 or epoch == 0:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss.item()}, Val: {val_loss.item()}"
                    )
                scheduler.step()

            torch.save(self.hypernetwork.state_dict(), self.save_path)

        return


def main():
    hypernetwork = HyperNetwork3D()

    inr_trainer = INRTrainer3D(debug=True)

    hypernetwork_trainer = HyperNetworkTrainer(
        hypernetwork,
        sMLP,
        inr_trainer,
        save_path="ADL4CV/models/3D/finalHypernetwork3D.pth",
        override=True,
    )

    armadillo_combined_idx = [
        54,
        61,
        201,
        226,
        260,
        279,
        341,
        344,
        370,
        403,
        457,
        489,
        522,
        539,
        558,
        597,
        55,
        138,
        473,
        574,
    ]

    armadillo_pairs = [
        [x, y] for x in armadillo_combined_idx for y in armadillo_combined_idx
    ]
    random.shuffle(armadillo_pairs)

    train_pairs = armadillo_pairs[:370]
    val_pairs = armadillo_pairs[370:]

    # hypernetwork_trainer.train_hypernetwork(train_pairs, val_pairs, False, False, 8)
    hypernetwork.load_state_dict(
        torch.load(
            "ADL4CV/models/3D/finalHypernetwork3D.pth",
            map_location=torch.device(device),
        )
    )

    with open("config.json") as json_file:
        json_file = json.load(json_file)
        INR_model_config = json_file["INR_model_config_3D"]

    model = sMLP(seed=42, INR_model_config=INR_model_config)
    step = 0.05
    scale_1, scale_2, scale_3 = 1, 2, 1
    for train_pair in train_pairs[:32]:
        idx_1, idx_2 = train_pair
        model.load_state_dict(
            torch.load(
                f"ADL4CV/data/model_data/shrec_16/T{idx_1}.pth",
                map_location=torch.device(device),
            )
        )
        params_1 = flatten_model_weights(model)
        model.load_state_dict(
            torch.load(
                f"ADL4CV/data/model_data/shrec_16/T{idx_2}.pth",
                map_location=torch.device(device),
            )
        )
        params_2 = flatten_model_weights(model)
        params_cat = torch.cat((params_1, params_2)).to(device)

        predicted_weights = hypernetwork(params_cat).unsqueeze(0)
        weights, biases = unflatten_weights(predicted_weights, model)

        render_and_store_models(
            step,
            scale_1,
            scale_2,
            scale_3,
            model_path=None,
            weights=weights,
            biases=biases,
            save_path=f"ADL4CV/evaluation/pairs/train_{idx_1}_{idx_2}.obj",
        )

    for val_pair in val_pairs:
        idx_1, idx_2 = val_pair
        model.load_state_dict(
            torch.load(
                f"ADL4CV/data/model_data/shrec_16/T{idx_1}.pth",
                map_location=torch.device(device),
            )
        )
        params_1 = flatten_model_weights(model)
        model.load_state_dict(
            torch.load(
                f"ADL4CV/data/model_data/shrec_16/T{idx_2}.pth",
                map_location=torch.device(device),
            )
        )
        params_2 = flatten_model_weights(model)
        params_cat = torch.cat((params_1, params_2)).to(device)

        predicted_weights = hypernetwork(params_cat).unsqueeze(0)
        weights, biases = unflatten_weights(predicted_weights, model)

        render_and_store_models(
            step,
            scale_2,
            scale_1,
            scale_2,
            model_path=None,
            weights=weights,
            biases=biases,
            save_path=f"ADL4CV/evaluation/pairs/val_{idx_1}_{idx_2}.obj",
        )


if __name__ == "__main__":
    main()
