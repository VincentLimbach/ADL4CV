import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pykdtree.kdtree import KDTree

from torchvision import datasets, transforms
from architectures import INR, sMLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class INRTrainer3D:
    def __init__(self, debug=False):
        self.debug = debug
        with open("config.json") as json_file:
            json_file = json.load(json_file)
            self.INR_model_config = json_file["INR_model_config_3D"]
            self.INR_trainer_config = json_file["INR_trainer_config_3D"]

    def fit_inr(self, dataloader, index, model_cls, seed, save=False, save_name=None):

        model = model_cls(seed=seed, INR_model_config=self.INR_model_config)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.INR_trainer_config["lr"], weight_decay=self.INR_trainer_config.get("weight_decay", 0))
        epochs = self.INR_trainer_config["epochs"]
        train_generator = iter(dataloader)

        for epoch in range(epochs):
            optimizer.zero_grad()

            try:
                # sampling_scheduler(step)
                model_input, gt = next(train_generator)
            except StopIteration:
                train_generator = iter(dataloader)
                model_input, gt = next(train_generator)

            model_input = dict2cuda(model_input)
            gt = dict2cuda(gt)

            model_output = model(model_input[0]).to(gt[0].device)

            loss = criterion(model_output, gt[0])
            loss.backward()
            optimizer.step()

            if self.debug and epoch % 50 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss}')
        torch.save(model.state_dict(), save_name)

        return model.state_dict()

    def init_dataloader(self, pointcloud_path, coarse_scale, fine_scale):
        ''' load sdf dataloader via eikonal equation or fitting sdf directly '''

        sdf_dataset = MeshSDF(pointcloud_path,
                            coarse_scale,
                            fine_scale)

        dataloader = DataLoader(sdf_dataset, shuffle=False,
                                batch_size=1)

        return dataloader 

    def fit_inrs(self, dataset, indices, model_cls, seed, save_path):
        for index in indices:
            train_dataloader = self.init_dataloader(f"./data/shrec_16/T{index}.xyz", 5*1e-1, 1e-3) #1e-2, 5*1e-3)

            final_model = self.fit_inr(train_dataloader, 0, model_cls, seed, save=True, save_name = save_path + str(index) + ".pth")


def dict2cuda(a_dict):
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp = value.to(device) #.cuda()})
        elif isinstance(value, dict):
            tmp = dict2cuda(value)
        elif isinstance(value, list) or isinstance(value, tuple):
            if isinstance(value[0], torch.Tensor):
                tmp = [v.to(device) for v in value] #.cuda() for v in value]})
        else:
            tmp = value
    return tmp

class MeshSDF(Dataset):
    ''' convert point cloud to SDF '''

    def __init__(self, pointcloud_path, coarse_scale=1e-1, fine_scale=1e-3):
        super().__init__()
        self.pointcloud_path = pointcloud_path
        self.coarse_scale = coarse_scale
        self.fine_scale = fine_scale

        self.load_mesh(pointcloud_path)

    def __len__(self):
        return 1000

    def load_mesh(self, pointcloud_path):
        pointcloud = np.genfromtxt(pointcloud_path)
        self.v = pointcloud[:, :3]
        self.n = pointcloud[:, 3:]

        n_norm = (np.linalg.norm(self.n, axis=-1)[:, None])
        n_norm[n_norm == 0] = 1.
        self.n = self.n / n_norm
        self.v = self.normalize(self.v)
        self.kd_tree = KDTree(self.v)
        print('loaded pc')

    def normalize(self, coords):
        coords -= np.mean(coords, axis=0, keepdims=True)
        coord_max = np.amax(coords)
        coord_min = np.amin(coords)
        coords = (coords - coord_min) / (coord_max - coord_min) * 0.9
        coords -= 0.45
        return coords

    def sample_surface(self):
        #idx = np.random.randint(0, self.v.shape[0], self.num_samples)
        points = self.v.copy() #[idx]
        points[::2] += np.random.laplace(scale=self.coarse_scale, size=(points.shape[0]//2, points.shape[-1]))
        points[1::2] += np.random.laplace(scale=self.fine_scale, size=(points.shape[0]//2, points.shape[-1]))
        # wrap around any points that are sampled out of bounds
        points[points > 0.5] -= 1
        points[points < -0.5] += 1

        # use KDTree to get distance to surface and estimate the normal
        sdf, idx = self.kd_tree.query(points, k=3)
        avg_normal = np.mean(self.n[idx], axis=1)
        sdf = np.sum((points - self.v[idx][:, 0]) * avg_normal, axis=-1)
        sdf = sdf[..., None]
        return points, sdf

    def __getitem__(self, idx):
        coords, sdf = self.sample_surface()

        return {'coords': torch.from_numpy(coords).float()}, \
               {'sdf': torch.from_numpy(sdf).float()}
