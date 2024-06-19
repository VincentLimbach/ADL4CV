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





