import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from dotenv import load_dotenv
from architectures import INR, sMLP
from path import Path
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class INRTrainer:
    def __init__(self, subdirectory, override=False, debug=False):
        load_dotenv()
        self.model_save_dir = Path("./data/INR" + subdirectory)
        self.setup_directory()
        self.override = override
        self.debug = debug

    def setup_directory(self):
        if not os.path.exists(Path("./data/INR")):
            os.makedirs(Path("./data/INR"))
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

    def fit_inr(self, dataset, index, model_cls, seed, arg_dict):

        img, _ = dataset[index]
        height, width = img.shape[1], img.shape[2]

        coords = [[i, j] for i in range(height) for j in range(width)]
        intensities = [img[:, i, j].item() for i in range(height) for j in range(width)]
        coords = torch.tensor(coords, dtype=torch.float32)
        intensities = torch.tensor(intensities, dtype=torch.float32)

        model = model_cls(seed=seed, arg_dict=arg_dict)

        model_save_path = Path(self.model_save_dir / Path(str(model) + f"_{index}.pth"))
        if (not self.override and os.path.exists(model_save_path)):
            return -1
            #raise ValueError("sMLP already exists")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 5000
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(coords)
            loss = criterion(outputs.squeeze(), intensities)
            loss.backward()
            optimizer.step()

            if self.debug and epoch % 50 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        torch.save(model.state_dict(), model_save_path)

        return loss.item()

    def fit_inrs(self, dataset, indices, model_cls, seed, arg_dict):
        losses = []
        for index in indices:
            final_loss = self.fit_inr(dataset, index, model_cls, seed, arg_dict)
            losses.append(final_loss)
        return losses


def test_smlp_on_mnist():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    arg_dict = {
        'input_feature_dim': 2,
        'output_feature_dim': 1,
        'hidden_features': 64,
        'layers': 2,
        'positional': True,
        'd_model': 16
    }

    trainer = INRTrainer(subdirectory='/sMLP', debug=True)

    indices = list(range(10))
    model_cls = sMLP
    num_models = len(indices)

    losses = trainer.fit_inrs(mnist_dataset, indices, model_cls, 42, arg_dict)

    print("Final Losses for each trained sMLP model on MNIST images:")
    for i, loss in zip(indices, losses):
        print(f"Model {i}: Loss = {loss}")

    index_to_check = 5
    model = model_cls(seed=42, arg_dict=arg_dict)
    model_path = trainer.model_save_dir / f"{str(model)}_{index_to_check}.pth"

    model.load_state_dict(torch.load(model_path))

    img = mnist_dataset[index_to_check][0]
    height, width = img.shape[1], img.shape[2]

    coords = [[i, j] for i in range(height) for j in range(width)]
    coords = torch.tensor(coords, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(coords).numpy()

    predictions = predictions.reshape((height, width))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(img.squeeze(), cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(predictions, cmap='gray')
    ax[1].set_title('Model Predictions')
    ax[1].axis('off')

    plt.show()

if __name__ == "__main__":
    test_smlp_on_mnist()