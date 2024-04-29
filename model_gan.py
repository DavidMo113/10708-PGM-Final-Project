# reference: https://github.com/Moeinh77/Simpson-face-generator-DCGAN-pytorch

import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):

    def __init__(self, noise_size):
        super(Generator, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(noise_size, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsample = nn.Sequential( # 256 * 4 * 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 128 * 8 * 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 64 * 16 * 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), # 3 * 32 * 32
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, x):
        p = self.project(x)
        p = p.view(-1, 256, 4, 4)
        return self.upsample(p)

    def sample(self, x):
        return self.forward(x)

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.downsample = nn.Sequential( # 3 * 32 * 32
            nn.Conv2d(3, 32, 4, 2, 1), # 32 * 16 * 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), # 64 * 8 * 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 128 * 4 * 4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), # 256 * 2 * 2
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 2, 1), # 1 * 1 * 1
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        out = self.downsample(x)
        return out.view(-1)