import argparse
import math
import os
import shutil

import imageio
import numpy as np

import torch
import torch.nn.functional as F

from data import get_dataloader
from model_dm import Unet, p_losses, sample

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='dm')
    parser.add_argument('--sample_dir', type=str, default='dm/sample')
    parser.add_argument('--ckpt_dir', type=str, default='dm/ckpt')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--noise_level', type=float, default=0)
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--timesteps', type=int, default=300)
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

def save_samples(args, model, epoch):
    generated_images = sample(model, image_size=args.image_size, batch_size=args.num_samples, channels=3)[-1]

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
    model = Unet(dim=args.image_size, channels=3, dim_mults=(1, 2, 4,)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # training / logging
    for epoch in range(args.num_epochs):
        # training
        training_loss = 0
        for x, _ in dataloader:
            x = x.to(device)

            optimizer.zero_grad()

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, args.timesteps, (len(x),), device=device).long()

            loss = p_losses(model, x, t, loss_type="huber")

            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        # logging
        training_loss /= len(dataloader)
        print(f'epoch [{epoch+1:04d}/{args.num_epochs:04d}] | loss: {training_loss:6.4f}')
        if (epoch+1) % args.logging_every == 0:
            # sample
            save_samples(args, model, epoch+1)
            save_images(args, next(iter(dataloader))[0], epoch+1)
            # ckpt
            torch.save(model.state_dict(), f'{args.ckpt_dir}/{epoch+1}.ckpt')

if __name__ == '__main__':
    main()