import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from ADL4CV.utils import flatten_model_weights


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class ObjectINRDataset(Dataset):
    def __init__(
        self,
        model_cls,
        trainer,
        model_save_dir,
        on_the_fly=False,
        path=None,
        device="cpu",
    ):
        self.model_cls = model_cls
        self.trainer = trainer
        self.on_the_fly = on_the_fly
        self.model_save_dir = model_save_dir
        self.path = path
        self.device = device

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        obj = np.genfromtxt(
            f"ADL4CV/data/gt_data/shrec_16_processed_collapsed/T{index}.xyz"
        )
        obj = torch.tensor(obj)
        obj = obj.to(self.device)
        model_path = os.path.join(self.model_save_dir, f"T{index}.pth")
        if self.path is not None:
            model_path = os.path.join(
                self.model_save_dir, self.path + str(index) + ".pth"
            )

        if os.path.exists(model_path):
            model = self.model_cls(
                seed=42, INR_model_config=self.trainer.INR_model_config
            ).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))

        return obj, torch.tensor([0]), flatten_model_weights(model)
