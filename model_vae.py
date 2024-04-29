# reference: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, noise_size):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential( # 3 * 32 * 32
            nn.Conv2d(3, 32, 3, 2, 1), # 32 * 16 * 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), # 64 * 8 * 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), # 128 * 4 * 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), # 256 * 2 * 2
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        self.fc_mu = nn.Linear(256*2*2, noise_size)
        self.fc_var = nn.Linear(256*2*2, noise_size)
        self.fc_dec = nn.Linear(noise_size, 256*2*2)

        self.decoder = nn.Sequential( # 256 * 2 * 2
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), # 128 * 4 * 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), # 64 * 8 * 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), # 32 * 16 * 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, 3, 2, 1, 1), # 32 * 32 * 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1), # 3 * 32 * 32
            nn.Tanh(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        z = eps * std + mu
        return z

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 256, 2, 2)
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def sample(self, z):
        return self.decode(z)