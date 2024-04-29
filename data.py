import torch
from torch.utils.data import Subset, DataLoader
import torchvision
import torchvision.transforms as transforms

def get_dataloader(batch_size, noise_level, cls='dog'):
    # dataset
    transform = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x + noise_level * torch.randn_like(x)),
        # transforms.Normalize((0.491, 0.482 ,0.447), (0.247, 0.243, 0.262)), # CIFAR10
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             transform=transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            transform=transform, download=True)

    cls_index = train_set.class_to_idx[cls]
    train_cls_indices = [i for i in range(len(train_set)) if train_set[i][1] == cls_index]
    test_cls_indices = [i for i in range(len(test_set)) if test_set[i][1] == cls_index]

    train_set = Subset(train_set, train_cls_indices)
    test_set = Subset(test_set, test_cls_indices)
    dataset = train_set + test_set
    print(f'# training data = {len(dataset)}')

    # dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader