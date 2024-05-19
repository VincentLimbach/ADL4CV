import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(2177, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 2177),
        )

        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

