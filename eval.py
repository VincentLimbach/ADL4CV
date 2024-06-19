from INRTrainer3D import INRTrainer3D
from torchvision import datasets, transforms
from architectures import sMLP
import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
#from HypernetworkTrainer import *
import json
import mcubes
import trimesh


trainer3D = INRTrainer3D(debug=True)
trainer3D.fit_inrs(42,[61],sMLP,42,"./data/shrec_16/MLPs/MLP_")
print("fitted")


with open("config.json") as json_file:
    json_file = json.load(json_file)
    INR_model_config = json_file["INR_model_config_3D"]
    INR_trainer_config = json_file["INR_trainer_config_3D"]

model = sMLP(42, INR_model_config)
model.load_state_dict(torch.load("./data/shrec_16/MLPs/MLP_61.pth", map_location=torch.device('cpu')))

step = .05

pts = torch.tensor(np.mgrid[-.5:.5:step, -5:.5:step, -.5:.5:step])
def apply_sdf(model, grid):
    _, dim_x, dim_y, dim_z = grid.shape
    res = np.zeros((dim_x, dim_y, dim_z), dtype='float32')
    for i in range(dim_x):
        for j in range(dim_y):
            for k in range(dim_z):
                res[i,j,k] = model(grid[:,i,j,k].unsqueeze(0).float())
    return res

sdf = apply_sdf(model, pts)

vertices, triangles = mcubes.marching_cubes(-sdf, 0)

mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
N = 2/step
mesh.vertices = (mesh.vertices / N - 0.5) + 0.5/N
mesh.export(f"./T61_sdf.obj")

ran = np.linspace(0,1, 50)
for x in ran:
    y = torch.tensor([[x,0,0]])
    print(model(y.float()))

