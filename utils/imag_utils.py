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


def generate_merged_image(triples):
    max_height = max(x_offset + img.shape[1] for img, x_offset, y_offset in triples)
    max_width = max(y_offset + img.shape[2] for img, x_offset, y_offset in triples)
    
    merged_img = torch.zeros((1, max_height, max_width), dtype=torch.float32)
    
    for img, x_offset, y_offset in triples:
        _, img_height, img_width = img.shape
        merged_img[:, x_offset:x_offset+img_height, y_offset:y_offset+img_width] = torch.max(
            merged_img[:, x_offset:x_offset+img_height, y_offset:y_offset+img_width], img)
    
    return merged_img


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=16):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.frequency = self.generate_frequencies()

    def generate_frequencies(self):
        frequencies = torch.zeros(self.d_model // 2)
        frequencies[0] = 0 
        period_length = 72
        for i in range(1, self.d_model // 2):
            frequencies[i] = 2 * math.pi / period_length
            period_length /= 2
        return frequencies

    def forward(self, x):
        batch_size, _ = x.shape

        x_enc = x[:, :1] * self.frequency
        y_enc = x[:, 1:] * self.frequency
        enc_x = torch.cat((torch.sin(x_enc), torch.cos(x_enc)), dim=-1)
        enc_y = torch.cat((torch.sin(y_enc), torch.cos(y_enc)), dim=-1)

        encoded_input = torch.cat((enc_x, enc_y), dim=-1)
        return encoded_input
