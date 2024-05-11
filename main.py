import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from architectures import sMLP, MLP, HyperNetwork

import matplotlib.pyplot as plt


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

def train_smaller_models(image, model_save_name, helper=0):
    if helper==0:
        image = F.pad(image, (0, 28, 0, 0))
    elif helper==1:
        image = F.pad(image, (28, 0, 0, 0))
    height, width = image.shape[1], image.shape[2]

    coords = [[i, j] for i in range(height) for j in range(width)]
    intensities = [image[:, i, j].item() for i in range(height) for j in range(width)]
    coords = torch.tensor(coords, dtype=torch.float32)
    intensities = torch.tensor(intensities, dtype=torch.float32)

    model = sMLP(32,64, 1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5000
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(coords)
        loss = criterion(outputs.squeeze(), intensities)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    
    torch.save(model.state_dict(), model_save_name)

    model.eval()
    with torch.no_grad():
        pred_intensities = model(coords)
        pred_image = pred_intensities.reshape(height, width).numpy()


def extract_and_concat_weights(model_paths, concat=False):
    concatenated_weights = []
    for path in model_paths:
        model = sMLP(32,64, 1)
        model.load_state_dict(torch.load(path))
        model_weights = []
        for param in model.parameters():
            model_weights.append(param.data.view(-1))
        concatenated_weights.append(torch.cat(model_weights))
    if not concat:
        return torch.stack((concatenated_weights))
    return torch.cat(concatenated_weights)



first_image, _ = train_dataset[0]
#train_smaller_models(first_image, 'model_1.pth')

second_image, _ = train_dataset[1]
#train_smaller_models(second_image, 'model_2.pth', 1)

concat_image = torch.cat([first_image, second_image], dim=2) 

#train_smaller_models(concat_image, 'model_3.pth', 2)

model_weights_input = extract_and_concat_weights(['model_1.pth', 'model_2.pth']) #, concat=False)
hypernet = HyperNetwork(input_dim=model_weights_input.numel())
hyper_optimizer = optim.Adam(hypernet.parameters(), lr=0.0001)
print(model_weights_input.shape)

coords = [[i, j] for i in range(28) for j in range(56)]
coords = torch.tensor(coords, dtype=torch.float32)
intensities = concat_image.view(-1).float() 

new_model = sMLP(32,64, 1)

writer = SummaryWriter('runs/hypernetwork_experiment')

epochs = 4000
for epoch in range(epochs):
    hyper_optimizer.zero_grad()
    predicted_weights = hypernet(model_weights_input)
    outputs = new_model(coords, predicted_weights)
    loss = F.mse_loss(outputs.squeeze(), intensities)
    loss.backward()

    hyper_optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

outputs = new_model(coords, predicted_weights)
predicted_image = outputs.view(28, 56).detach().numpy()

fig, axes = plt.subplots(nrows=2,figsize=(8,4))
axes[0].imshow(predicted_image, cmap='gray')
axes[0].set_title("predicted")
axes[1].imshow(concat_image[0], cmap="gray")
axes[1].set_title("original")
fig.tight_layout()
plt.show()

def load_weights_and_predict(model_path, coords):
    model = sMLP(32,64, 1) 
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        predicted_outputs = model(coords)
    predicted_image = predicted_outputs.view(28, 56).detach().numpy()
    return predicted_image

model_paths = ['model_1.pth', 'model_2.pth']
titles = ['Predicted Image from Model 1', 'Predicted Image from Model 2']

#fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#for ax, model_path, title in zip(axes, model_paths, titles):
#    predicted_image = load_weights_and_predict(model_path, coords)
#    ax.imshow(predicted_image, cmap='gray')
#    ax.set_title(title)
#    ax.axis('off') 
#plt.show()

originals = [first_image[0], second_image[0]]

#fig, axes = plt.subplots(2, 2, figsize=(10, 5))
#for idx, model_path, title in zip([0,1], model_paths, titles):
#    predicted_image = load_weights_and_predict(model_path, coords)
#    axes[idx,0].imshow(predicted_image, cmap='gray')
#    axes[idx,1].imshow(originals[idx], cmap="gray")
#    axes[idx,0].set_title(title)
#    axes[idx,1].set_title("original")
#    #axes.axis('off') 
#fig.tight_layout()
#plt.show()

