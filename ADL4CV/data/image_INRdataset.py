import os
import sys
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from ADL4CV.architectures import sMLP
from ADL4CV.data.INR_trainer2D import INRTrainer2D
from ADL4CV.utils import flatten_model_weights

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ImageINRDataset(Dataset):
    def __init__(self, dataset_name, model_cls, trainer, model_save_dir, on_the_fly=False, path=None, device='cpu'):
        self.dataset_name = dataset_name
        self.model_cls = model_cls
        self.trainer = trainer
        self.on_the_fly = on_the_fly
        self.model_save_dir = model_save_dir
        self.path = path
        self.device = device

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        if self.dataset_name == "MNIST":
            self.dataset = datasets.MNIST(root='ADL4CV/data/gt_data/', train=True, download=True, transform=transforms.ToTensor())
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = img.to(self.device)  # Move the image to the specified device
        label = torch.tensor(label).to(self.device)  # Move the label to the specified device
        model_path = os.path.join(self.model_save_dir, f"sMLP_{index}.pth")
        if self.path is not None:
            model_path = os.path.join(self.model_save_dir, self.path + str(index) + ".pth")

        if os.path.exists(model_path):
            model = self.model_cls(seed=42, INR_model_config=self.trainer.INR_model_config).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            if self.on_the_fly:
                state_dict = self.trainer.fit_inr(self.dataset, index, self.model_cls, 42)
                model = self.model_cls(seed=42, INR_model_config=self.trainer.INR_model_config).to(self.device)
                model.load_state_dict(state_dict)
                torch.save(state_dict, model_path)
            else:
                raise Exception(f"Model for index {index} not found and on_the_fly is set to False.")

        return img, label, flatten_model_weights(model)

def main():
    inr_trainer = INRTrainer2D(debug=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_inr_dataset = ImageINRDataset("MNIST", sMLP, inr_trainer, "ADL4CV/data/model_data/MNIST", on_the_fly=True, device=device)

    for index in range(11000, 16384):
        try:
            img, label, model_weights = img_inr_dataset[index]
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()
