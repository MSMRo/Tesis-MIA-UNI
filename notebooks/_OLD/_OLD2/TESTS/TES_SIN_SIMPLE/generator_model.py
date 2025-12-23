import torch
import torch.nn as nn

class GeneratorGAN(nn.Module):
    def __init__(self, noise_dim=100, output_dim=1000):
        super(GeneratorGAN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
