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

import torch

def generate_merged_image(img_1_batch, img_2_batch, x_1_offsets, y_1_offsets, x_2_offsets, y_2_offsets):
    device = img_1_batch.device
    x_1_offsets = x_1_offsets.to(device)
    y_1_offsets = y_1_offsets.to(device)
    x_2_offsets = x_2_offsets.to(device)
    y_2_offsets = y_2_offsets.to(device)
    
    batch_size, img_height, img_width = img_1_batch.shape
    img_2_height, img_2_width = img_2_batch.shape[1], img_2_batch.shape[2]
    
    max_height = max(img_height + y_1_offsets.max().item(), img_height + y_2_offsets.max().item())
    max_width = max(img_width + x_1_offsets.max().item(), img_width + x_2_offsets.max().item())
    
    merged_imgs = torch.zeros((batch_size, max_height, max_width), dtype=torch.float32, device=device)
    
    batch_indices = torch.arange(batch_size, device=device)
    
    y1_indices = y_1_offsets[:, None, None] + torch.arange(img_height, device=device)[None, :, None]
    x1_indices = x_1_offsets[:, None, None] + torch.arange(img_width, device=device)[None, None, :]
    
    merged_imgs[batch_indices[:, None, None], y1_indices, x1_indices] = img_1_batch
    
    y2_indices = y_2_offsets[:, None, None] + torch.arange(img_2_height, device=device)[None, :, None]
    x2_indices = x_2_offsets[:, None, None] + torch.arange(img_2_width, device=device)[None, None, :]
    
    merged_imgs[batch_indices[:, None, None], y2_indices, x2_indices] = torch.max(
        merged_imgs[batch_indices[:, None, None], y2_indices, x2_indices],
        img_2_batch
    )
    
    return merged_imgs



