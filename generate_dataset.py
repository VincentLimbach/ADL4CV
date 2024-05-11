from architectures import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def train_inr(image, model_save_name):
    height, width = image.shape[1], image.shape[2]

    coords = [[i, j] for i in range(height) for j in range(width)]
    intensities = [image[:, i, j].item() for i in range(height) for j in range(width)]
    coords = torch.tensor(coords, dtype=torch.float32)
    intensities = torch.tensor(intensities, dtype=torch.float32)

    model = sMLP(32,64, 1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(coords)
        loss = criterion(outputs.squeeze(), intensities)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        if epoch == epochs-1:
                print(loss)
                final_loss = loss

    torch.save(model.state_dict(), f"./data/INRs/{model_save_name}")

    model.eval()
    with torch.no_grad():
        pred_intensities = model(coords)
        pred_image = pred_intensities.reshape(height, width).numpy()
    
    return final_loss

        

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

losses = []
for i in range(2000):
    print(f"INR_{i}.pth")
    final_loss = train_inr(train_dataset[i][0], f"INR_{i}.pth")
    losses.append(final_loss)

print(losses)
np.savetxt(losses)

model_paths = ['./data/INRs/INR_1.pth', './data/INRs/INR_2.pth']
titles = ['Predicted Image from Model 1', 'Predicted Image from Model 2']
coords = [[i, j] for i in range(28) for j in range(28)]
coords = torch.tensor(coords, dtype=torch.float32)
originals = [train_dataset[0][0][0], train_dataset[1][0]][0]


def load_weights_and_predict(model_path, coords):
    model = sMLP(32,64, 1) 
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        predicted_outputs = model(coords)
    predicted_image = predicted_outputs.view(28, 28).detach().numpy()
    return predicted_image


fig, axes = plt.subplots(2, 2, figsize=(10, 5))
for idx, model_path, title in zip([0,1], model_paths, titles):
    predicted_image = load_weights_and_predict(model_path, coords)
    axes[idx,0].imshow(predicted_image, cmap='gray')
    axes[idx,1].imshow(originals[idx], cmap="gray")
    axes[idx,0].set_title(title)
    axes[idx,1].set_title("original")
    #axes.axis('off') 
fig.tight_layout()
plt.show()

