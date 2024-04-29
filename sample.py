import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import Subset

from model_vae import VAE
from model_gan import Generator
from model_dm import Unet, sample

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fid_dir', type=str, default='fid')
    parser.add_argument('--real_img_dir', type=str, default='fid/real')
    parser.add_argument('--fake_img_dir', type=str, default='fid/fake')
    parser.add_argument('--noise_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_samples', type=int, default=6000)
    args = parser.parse_args()
    return args

def main():
    # args
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.fid_dir, exist_ok=True)
    os.makedirs(args.real_img_dir, exist_ok=True)
    os.makedirs(args.fake_img_dir, exist_ok=True)

    # real data
    len_real_img_dir = len(os.listdir(args.real_img_dir))
    if len_real_img_dir == 0:
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

        cls_index = train_set.class_to_idx['dog']
        train_cls_indices = [i for i in range(len(train_set)) if train_set[i][1] == cls_index]
        test_cls_indices = [i for i in range(len(test_set)) if test_set[i][1] == cls_index]

        train_set = Subset(train_set, train_cls_indices)
        test_set = Subset(test_set, test_cls_indices)
        dataset = train_set + test_set
        print(f'# training data = {len(dataset)}')

        for i, (x, _) in enumerate(dataset):
            x.save(f'{args.real_img_dir}/{i+1}.png')
    else:
        print(f'already saved {len_real_img_dir} images')

    # fake data
    for model_name in ['vae', 'gan', 'dm']:
        for noise_level in [0, 0.1, 0.3, 0.5]:
            # directory
            model_name_dir = f'{args.fake_img_dir}/{model_name}_{noise_level}'
            os.makedirs(model_name_dir, exist_ok=True)
            if len(os.listdir(model_name_dir)) != 0:
                continue

            # model
            print(model_name, noise_level)
            if model_name == 'vae':
                model = VAE(args.noise_size).to(device)
                ckpt = torch.load(f'res_{noise_level}/{model_name}/ckpt/1000.ckpt')
            elif model_name == 'gan':
                model = Generator(noise_size=args.noise_size).to(device)
                ckpt = torch.load(f'res_{noise_level}/{model_name}/ckpt/G_1000.ckpt')
            elif model_name == 'dm':
                model = Unet(dim=args.image_size, channels=3, dim_mults=(1, 2, 4,)).to(device)
                ckpt = torch.load(f'res_{noise_level}/{model_name}/ckpt/1000.ckpt')
            model.load_state_dict(ckpt)

            # sample and save
            for i in tqdm(range(0, args.num_samples, args.batch_size)):
                fixed_noise = torch.randn(args.batch_size, args.noise_size).to(device)
                if model_name == 'vae' or model_name == 'gan':
                    generated_images = model.sample(fixed_noise)
                    generated_images = generated_images.cpu().detach().numpy()
                elif model_name == 'dm':
                    generated_images = sample(model, image_size=args.image_size, batch_size=args.batch_size, channels=3)[-1]
                for j in range(i, min(i+args.batch_size, args.num_samples)):
                    generated_image = generated_images[j-i].transpose(1, 2, 0)
                    generated_image = np.uint8(255 * (generated_image + 1) / 2)
                    generated_image = Image.fromarray(generated_image)
                    generated_image.save(f'{model_name_dir}/{j+1}.png')
            print(f'saved {len(os.listdir(model_name_dir))} images')

if __name__ == '__main__':
    main()