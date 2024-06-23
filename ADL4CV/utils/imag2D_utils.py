import torch
import torch.nn as nn 
import math
import matplotlib.pyplot as plt
import numpy as np

def plot_tensor_as_grayscale(tensor, title):
    np_img = tensor.numpy()
    plt.imshow(np_img, cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.show()

def generate_coordinates(img):
    _, h, w = img.shape
    coordinates = [(x, y) for x in range(w) for y in range(h)]
    return torch.tensor(coordinates, dtype=torch.float32)

def generate_merged_image(img_1_batch, img_2_batch, x_1_offsets, y_1_offsets, x_2_offsets, y_2_offsets, device):
    batch_size, img_height, img_width = img_1_batch.shape
    img_2_height, img_2_width = img_2_batch.shape[1], img_2_batch.shape[2]
    
    max_height = max(img_height + y_1_offsets.max().item(), img_2_height + y_2_offsets.max().item())
    max_width = max(img_width + x_1_offsets.max().item(), img_2_width + x_2_offsets.max().item())
    
    merged_imgs = torch.zeros((batch_size, max_height, max_width), dtype=torch.float32, device=device)
    
    batch_indices = torch.arange(batch_size, device=device)[:, None, None]

    y1_indices = y_1_offsets[:, None] + torch.arange(img_height, device=device)
    x1_indices = x_1_offsets[:, None] + torch.arange(img_width, device=device)
    
    y2_indices = y_2_offsets[:, None] + torch.arange(img_2_height, device=device)
    x2_indices = x_2_offsets[:, None] + torch.arange(img_2_width, device=device)

    merged_imgs[batch_indices, y1_indices[:, :, None], x1_indices[:, None, :]] = img_1_batch

    merged_imgs[batch_indices, y2_indices[:, :, None], x2_indices[:, None, :]] = torch.max(
        merged_imgs[batch_indices, y2_indices[:, :, None], x2_indices[:, None, :]],
        img_2_batch
    )
    
    return merged_imgs

import torch
import torch.nn as nn 
import math
import matplotlib.pyplot as plt
import numpy as np
from pykdtree.kdtree import KDTree


def plot_tensor_as_grayscale(tensor, title):
    np_img = tensor.numpy()
    plt.imshow(np_img, cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.show()

def generate_coordinates(img):
    _, h, w = img.shape
    coordinates = [(x, y) for x in range(w) for y in range(h)]
    return torch.tensor(coordinates, dtype=torch.float32)

def generate_merged_image(img_1_batch, img_2_batch, x_1_offsets, y_1_offsets, x_2_offsets, y_2_offsets, device):
    batch_size, img_height, img_width = img_1_batch.shape
    img_2_height, img_2_width = img_2_batch.shape[1], img_2_batch.shape[2]
    
    max_height = max(img_height + y_1_offsets.max().item(), img_2_height + y_2_offsets.max().item())
    max_width = max(img_width + x_1_offsets.max().item(), img_2_width + x_2_offsets.max().item())
    
    merged_imgs = torch.zeros((batch_size, max_height, max_width), dtype=torch.float32, device=device)
    
    batch_indices = torch.arange(batch_size, device=device)[:, None, None]

    y1_indices = y_1_offsets[:, None] + torch.arange(img_height, device=device)
    x1_indices = x_1_offsets[:, None] + torch.arange(img_width, device=device)
    
    y2_indices = y_2_offsets[:, None] + torch.arange(img_2_height, device=device)
    x2_indices = x_2_offsets[:, None] + torch.arange(img_2_width, device=device)

    merged_imgs[batch_indices, y1_indices[:, :, None], x1_indices[:, None, :]] = img_1_batch

    merged_imgs[batch_indices, y2_indices[:, :, None], x2_indices[:, None, :]] = torch.max(
        merged_imgs[batch_indices, y2_indices[:, :, None], x2_indices[:, None, :]],
        img_2_batch
    )
    
    return merged_imgs

def normalize_batch(coords):
    coords -= torch.mean(coords, axis=1, keepdims=True)
    coord_max = torch.amax(coords, dim=(1,2))[:, None, None]
    coord_min = torch.amin(coords, dim=(1,2))[:, None, None]
    coords = (coords - coord_min) / (coord_max - coord_min) * 0.9
    coords -= 0.45
    return coords

def generate_merged_object(obj_1_batch, obj_2_batch, x_1_offsets, y_1_offsets, z_1_offsets, x_2_offsets, y_2_offsets, z_2_offsets, device):
    batch_size, _, _ = obj_1_batch.shape
    v_1_batch = obj_1_batch[:, :, :3]
    n_1_batch = obj_1_batch[:, :, 3:]
    v_2_batch = obj_2_batch[:, :, :3]
    n_2_batch = obj_2_batch[:, :, 3:]

    n_norm_1 = torch.linalg.norm(n_1_batch, dim=-1, keepdim=True)
    n_norm_1[n_norm_1 == 0] = 1.
    n_1_batch = n_1_batch / n_norm_1
    offsets_1 = torch.tensor([x_1_offsets, y_1_offsets, z_1_offsets], device=device).unsqueeze(0).unsqueeze(0)
    v_1_batch = normalize_batch(v_1_batch) + offsets_1

    n_norm_2 = torch.linalg.norm(n_2_batch, dim=-1, keepdim=True)
    n_norm_2[n_norm_2 == 0] = 1.
    n_2_batch = n_2_batch / n_norm_2
    offsets_2 = torch.tensor([x_2_offsets, y_2_offsets, z_2_offsets], device=device).unsqueeze(0).unsqueeze(0)
    v_2_batch = normalize_batch(v_2_batch) + offsets_2
    
    obj_1_batch[:, :, :3] = v_1_batch
    obj_1_batch[:, :, 3:] = n_1_batch
    obj_2_batch[:, :, :3] = v_2_batch
    obj_2_batch[:, :, 3:] = n_2_batch

    batch_combined = torch.cat((obj_1_batch, obj_2_batch), dim=1)
    v_batch = batch_combined[:, :, :3]
    n_batch = batch_combined[:, :, 3:]

    kd_trees = [KDTree(np.array(v_batch[i])) for i in range(batch_size)] # kriege das nicht gebatched weil wir für jedes pair einen eigenen tree brauchen

    points = np.array(v_batch)
    points[:, ::2] += np.random.laplace(scale=1e-1, size=(batch_size, points.shape[1]//2, points.shape[-1]))
    points[:, 1::2] += np.random.laplace(scale=1e-3, size=(batch_size, points.shape[1]//2, points.shape[-1]))

    sdfs = []
    for i in range(batch_size): #selbes problem wie oben mit dem tree
        sdf, idx = kd_trees[i].query(points[i], k=3)
        avg_normal = np.mean(np.array(n_batch)[i, idx], axis=1)
        sdf = np.sum((points[i] - np.array(v_batch)[i, idx][:, 0]) * avg_normal, axis=-1)
        sdf = sdf[..., None]
        sdfs.append(sdf)
    
    sdfs = np.stack(sdfs)
    
    return {'coords': torch.from_numpy(points).float()}, \
               {'sdf': torch.from_numpy(sdfs).float()}


