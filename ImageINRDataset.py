import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from architectures import sMLP
from INRTrainer import INRTrainer
from utils import flatten_model_weights

class ImageINRDataset(Dataset):
    def __init__(self, dataset_name, model_cls, arg_dict, trainer, on_the_fly=False):
        self.dataset_name = dataset_name
        self.model_cls = model_cls
        self.arg_dict = arg_dict
        self.trainer = trainer
        self.on_the_fly = on_the_fly

        if self.dataset_name == "MNIST":
            self.dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index, flat_weights=True):
        img, _ = self.dataset[index]

        model_path = self.trainer.model_save_dir / f"sMLP500_{index}" #f"sMLP_{index}.pth" #uncomment to use regular sMLP
        print(model_path)
        if model_path.exists():
            model = self.model_cls(seed=42, arg_dict=self.arg_dict)
            model.load_state_dict(torch.load(model_path))
        else:
            if self.on_the_fly:
                self.trainer.fit_inr(self.dataset, index, self.model_cls, 42, self.arg_dict)
                model = self.model_cls(seed=42, arg_dict=self.arg_dict)
                model.load_state_dict(torch.load(model_path))
            else:
                raise Exception(f"Model for index {index} not found and on_the_fly is set to False.")

        if flat_weights:
            return img, flatten_model_weights(model)
        else:
            return img, model

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

    custom_dataset = ImageINRDataset("MNIST", sMLP, arg_dict, trainer, on_the_fly=True)

    for index in range(20):
        try:
            img, model = custom_dataset[index]
            print(f"Successfully retrieved model for index {index}")
        except Exception as e:
            print(e)
