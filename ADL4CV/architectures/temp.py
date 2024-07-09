import torch
import torch.nn as nn

class SharpNet(nn.Module):
    def __init__(self):
        super(SharpNet, self).__init__()

        self.relu = nn.ReLU()
        self.W_org = torch.ones((72,72)).squeeze(0)
        self.W_sharp = torch.zeros((72,72)).squeeze(0)
        self.conv1 = nn.Conv2d(1, 8, 3, padding="same")
        self.conv2 = nn.Conv2d(8, 16, 3, padding="same")
        self.conv3 = nn.Conv2d(16, 1, 3, padding="same")
        print(sum(p.numel() for p in self.parameters()))

    def forward(self, x_org):
        x = x_org.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x_sharp = self.relu(self.conv3(x))
        return x_org * self.W_org+ x_sharp.squeeze(1) * self.W_sharp  

test = SharpNet()
x = torch.rand((512, 72,72))
x = test(x)
print(x.shape)
