import torch
from chamferdist import ChamferDistance
import numpy as np
from ADL4CV.utils import *
import random
from ADL4CV.architectures import *

device = "cpu"


def normalize(coords):
    coords -= np.mean(coords, axis=0, keepdims=True)
    coord_max = np.amax(coords)
    coord_min = np.amin(coords)
    coords = (coords - coord_min) / (coord_max - coord_min) * 0.9
    coords -= 0.45
    return coords


with open("config.json") as file:
    model = sMLP(
        seed=42, INR_model_config=json.load(file)["INR_model_config_3D"], device=device
    )


def extract_vertices(idx_1, idx_2, hypernetwork):
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

    scale_x = scale_z = 1
    scale_y = 2
    step = 0.05

    dimensions_x = int(2 * scale_x / step)
    dimensions_y = int(2 * scale_y / step)
    dimensions_z = int(2 * scale_z / step)

    coordinates = generate_coordinates(step, scale_x, scale_y, scale_z)

    sdf = apply_sdf(
        model, dimensions_x, dimensions_y, dimensions_z, coordinates, weights, biases
    )

    vertices, triangles = mcubes.marching_cubes(-sdf, 0)

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    N = scale_x / step
    print(len(mesh.vertices))
    mesh.vertices = mesh.vertices / N - np.array([[1, 2, 1]])

    return mesh.vertices


chamferDist = ChamferDistance()

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
random.seed(42)
random.shuffle(armadillo_pairs)

train_pairs = armadillo_pairs[:370]
val_pairs = armadillo_pairs[370:]

hypernetwork = HyperNetwork3D()
hypernetwork.load_state_dict(
    torch.load(
        "ADL4CV/models/3D/hypernetwork_armadillo_Leon.pth",
        map_location=torch.device(device),
    )
)

chamfer_train = []
for pair in train_pairs:
    idx_1, idx_2 = pair
    pointcloud_1 = np.genfromtxt(
        f"ADL4CV/data/gt_data/shrec_16_processed_collapsed/T{idx_1}.xyz"
    )
    pointcloud_2 = np.genfromtxt(
        f"ADL4CV/data/gt_data/shrec_16_processed_collapsed/T{idx_2}.xyz"
    )
    v_1 = pointcloud_1[:, :3]
    v_2 = pointcloud_2[:, :3]

    v_1 = normalize(v_1) + np.array([[0, -0.5, 0]])
    v_2 = normalize(v_2) + np.array([[0, 0.5, 0]])
    gt_vertices = np.vstack((v_1, v_2))
    gt_vertices = torch.tensor(gt_vertices).unsqueeze(0).float()

    merged_vertices = extract_vertices(idx_1, idx_2, hypernetwork)
    merged_vertices = torch.tensor(merged_vertices).unsqueeze(0).float()

    dist_forward = chamferDist(gt_vertices, merged_vertices, reverse=False)
    dist_backward = chamferDist(merged_vertices, gt_vertices, reverse=False)

    n_gt = gt_vertices.shape[1]
    n_merge = merged_vertices.shape[1]

    chamfer_dist = dist_forward / n_gt + dist_backward / n_merge
    # print(chamfer_dist, dist_forward, dist_backward)
    chamfer_train.append(chamfer_dist)

print(sum(chamfer_train) / len(chamfer_train))

chamfer_val = []
for pair in val_pairs:
    idx_1, idx_2 = pair
    pointcloud_1 = np.genfromtxt(
        f"ADL4CV/data/gt_data/shrec_16_processed_collapsed/T{idx_1}.xyz"
    )
    pointcloud_2 = np.genfromtxt(
        f"ADL4CV/data/gt_data/shrec_16_processed_collapsed/T{idx_2}.xyz"
    )
    v_1 = pointcloud_1[:, :3]
    v_2 = pointcloud_2[:, :3]

    v_1 = normalize(v_1) + np.array([[0, -0.5, 0]])
    v_2 = normalize(v_2) + np.array([[0, 0.5, 0]])
    gt_vertices = np.vstack((v_1, v_2))
    gt_vertices = torch.tensor(gt_vertices).unsqueeze(0).float()

    merged_vertices = extract_vertices(idx_1, idx_2, hypernetwork)
    merged_vertices = torch.tensor(merged_vertices).unsqueeze(0).float()

    dist_forward = chamferDist(gt_vertices, merged_vertices, reverse=False)
    dist_backward = chamferDist(merged_vertices, gt_vertices, reverse=False)

    n_gt = gt_vertices.shape[1]
    n_merge = merged_vertices.shape[1]

    chamfer_dist = dist_forward / n_gt + dist_backward / n_merge

    # print(chamfer_dist, dist_forward, dist_backward)
    chamfer_val.append(chamfer_dist)

print(sum(chamfer_val) / len(chamfer_val))
