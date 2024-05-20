import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from architectures import sMLP
from INRTrainer import INRTrainer
from utils import flatten_model_weights
import json

class ImageINRDataset(Dataset):
    def __init__(self, dataset_name, model_cls, trainer, model_save_dir, on_the_fly=False):
        self.dataset_name = dataset_name
        self.model_cls = model_cls
        self.trainer = trainer
        self.on_the_fly = on_the_fly
        self.model_save_dir = model_save_dir

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        if self.dataset_name == "MNIST":
            self.dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        #Should be dynamic based on class
        model_path = os.path.join(self.model_save_dir, f"sMLP_{index}.pth")
        if os.path.exists(model_path):
            model = self.model_cls(seed=42, INR_model_config=self.trainer.INR_model_config)
            model.load_state_dict(torch.load(model_path))
        else:
            if self.on_the_fly:
                state_dict = self.trainer.fit_inr(self.dataset, index, self.model_cls, 42)
                model = self.model_cls(seed=42, INR_model_config=self.trainer.INR_model_config)
                model.load_state_dict(state_dict)
                torch.save(state_dict, model_path)
            else:
                raise Exception(f"Model for index {index} not found and on_the_fly is set to False.")

        return img, flatten_model_weights(model)

def main():
    with open("config.json") as json_file:
        INR_model_config = json.load(json_file)["INR_model_config"]

    inr_trainer = INRTrainer(debug=True)
    img_inr_dataset = ImageINRDataset("MNIST", sMLP, inr_trainer, "data/INR/sMLP", on_the_fly=True)

    for index in range(30):
        try:
            img, model = img_inr_dataset[index]
            print(f"Successfully retrieved model for index {index}")
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()
