# generator_cgan.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        input_dim = latent_dim + num_classes  # z + one-hot

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True)
        )

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, y):
        z_cond = torch.cat((z, y), dim=1)
        x = self.fc(z_cond)
        x = x.view(-1, 256, 7, 7)
        img = self.conv_blocks(x)
        return img
