from INRTrainer import INRTrainer
from torchvision import datasets, transforms
from architectures import sMLP
import numpy as np
import torch
from matplotlib import pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#trainer = INRTrainer(debug=True)
#print(trainer.fit_inrs(dataset, np.arange(10)+1, sMLP, 42))
dic = {
        "input_feature_dim": 2,
        "output_feature_dim": 1,
        "hidden_features": 64,
        "layers": 2,
        "positional": True,
        "d_model": 16
    }
new_model = sMLP(42, dic)
coords = [[i, j] for i in range(28) for j in range(28)]
coords = torch.tensor(coords, dtype=torch.float32)

new_model.load_state_dict(torch.load("./data/INR/sMLP_comparison/sMLP5000_8"))
new_model.eval()
with torch.no_grad():
    predicted_outputs = new_model(coords)
predicted_image = predicted_outputs.view(28, 28).detach().numpy()

original=dataset[8]

fig, axes = plt.subplots(nrows=2,figsize=(8,4))
axes[0].imshow(predicted_image, cmap='gray')
axes[0].set_title("predicted")
axes[1].imshow(original[0][0], cmap="gray")
axes[1].set_title("original")
fig.tight_layout()
plt.show()