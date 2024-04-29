import argparse
import math
import os
import shutil

import imageio
import numpy as np

import torch

from data import get_dataloader
from model_gan import Generator, Discriminator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='gan')
    parser.add_argument('--sample_dir', type=str, default='gan/sample')
    parser.add_argument('--ckpt_dir', type=str, default='gan/ckpt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--noise_level', type=float, default=0)
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--noise_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--logging_every', type=int, default=100)
    args = parser.parse_args()
    return args

def create_image_grid(array, ncols=None):
    num_images, channels, cell_h, cell_w = array.shape

    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(math.floor(num_images / float(ncols)))
    result = np.zeros(
        (cell_h * nrows, cell_w * ncols, channels),
        dtype=array.dtype
    )
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[
                i * cell_h:(i + 1) * cell_h,
                j * cell_w:(j + 1) * cell_w, :
            ] = array[i * ncols + j].transpose(1, 2, 0)

    if channels == 1:
        result = result.squeeze()
    return result

def save_samples(args, model, fixed_noise, epoch):
    generated_images = model.sample(fixed_noise)
    generated_images = generated_images.cpu().detach().numpy()

    grid = create_image_grid(generated_images)
    grid = np.uint8(255 * (grid + 1) / 2)

    imageio.imwrite(f'{args.sample_dir}/{epoch}.png', grid)

def save_images(args, real_images, epoch):
    real_images = real_images.cpu().detach().numpy()

    grid = create_image_grid(real_images)
    grid = np.uint8(255 * (grid + 1) / 2)

    imageio.imwrite(f'{args.sample_dir}/real_{epoch}.png', grid)

def main():
    # args
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(args.root_dir):
        shutil.rmtree(args.root_dir)
    os.makedirs(args.root_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # data
    dataloader = get_dataloader(args.batch_size, args.noise_level)

    # model / optimizer
    G = Generator(noise_size=args.noise_size).to(device)
    D = Discriminator().to(device)

    G_optimizer = torch.optim.Adam(G.parameters(), args.lr, [args.beta1, args.beta2])
    D_optimizer = torch.optim.Adam(D.parameters(), args.lr, [args.beta1, args.beta2])

    # sample
    fixed_noise = torch.randn(args.num_samples, args.noise_size).to(device)
    torch.randn(args.num_samples, args.noise_size)

    # training / logging
    for epoch in range(args.num_epochs):
        # training
        D_training_loss = 0
        G_training_loss = 0
        for x, _ in dataloader:
            real_images = x.to(device)

            # D
            D_real_loss = torch.mean((D(real_images) - 1) ** 2)
            noise = torch.randn(args.batch_size, args.noise_size).to(device)
            fake_images = G(noise)
            D_fake_loss = torch.mean((D(fake_images.detach())) ** 2)
            D_total_loss = (D_real_loss + D_fake_loss) / 2
            D_optimizer.zero_grad()
            D_total_loss.backward()
            D_optimizer.step()

            # G
            noise = torch.randn(args.batch_size, args.noise_size).to(device)
            fake_images = G(noise)
            G_loss = torch.mean((D(fake_images) - 1) ** 2)
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            D_training_loss += D_total_loss.item()
            G_training_loss += G_loss.item()

        # logging
        D_training_loss /= len(dataloader)
        G_training_loss /= len(dataloader)
        print(f'epoch [{epoch+1:03d}/{args.num_epochs:03d}] | D loss: {D_training_loss:6.4f} | G loss: {G_training_loss:6.4f}')
        if (epoch+1) % args.logging_every == 0:
            # sample
            save_samples(args, G, fixed_noise, epoch+1)
            save_images(args, next(iter(dataloader))[0], epoch+1)
            # ckpt
            torch.save(D.state_dict(), f'{args.ckpt_dir}/D_{epoch+1}.ckpt')
            torch.save(G.state_dict(), f'{args.ckpt_dir}/G_{epoch+1}.ckpt')

if __name__ == '__main__':
    main()