import os

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
from torchvision import datasets, transforms


def get_data_loader(dataset_name, batch_size=512):
    if "cifar" in dataset_name:
        train_transform, test_transform = get_CIFAR_transforms()
        if "100" in dataset_name:
            return get_CIFAR_100_data_loader(train_transform, test_transform, batch_size)
        elif "10" in dataset_name:
            return get_CIFAR_data_loader(train_transform, test_transform, batch_size)
    elif dataset_name == "imagenet":
        train_transform, test_transform = get_imagenet_transforms()
        return get_ImageNet_dataloader(train_transform, test_transform, batch_size)
    else:
        raise ValueError("dataset name not supported")


def get_CIFAR_100_data_loader(train_transform, test_transform, batch_size=512):
    # get 10 random classes from CIFAR100
    classes = torch.randperm(100)[:10].tolist()
    print("loading CIFAR100 data")
    train_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, transform=train_transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, transform=test_transform, download=True
    )
    class_mapping = {original: new for new, original in enumerate(classes)}

    # Get indices of desired classes
    train_indices = [i for i in range(len(train_dataset)) if train_dataset.targets[i] in classes]
    test_indices = [i for i in range(len(test_dataset)) if test_dataset.targets[i] in classes]

    # Update the targets in train_dataset and test_dataset to the new labels
    for i in train_indices:
        train_dataset.targets[i] = class_mapping[train_dataset.targets[i]]
    for i in test_indices:
        test_dataset.targets[i] = class_mapping[test_dataset.targets[i]]

    # Create Subset objects with desired indices
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader


def get_CIFAR_data_loader(train_transform, test_trainsform, batch_size=512):
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, transform=train_transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=test_trainsform, download=True
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def get_ImageNet_dataloader(train_transform, test_trainsform, batch_size=512):
    train, test = get_ImageNet_dataset(train_transform, test_trainsform)
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


def get_ImageNet_dataset(train_transform, test_trainsform):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(cur_dir, "data/ImageNet")
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=test_trainsform)
    # I dont want to use all train_dataset for training, so I use SubsetRandomSampler
    # create random 10000 indices for Subset
    subset_indices = torch.randperm(len(train_dataset))[:10000]

    train_dataset = Subset(train_dataset, subset_indices)
    return train_dataset, val_dataset


def get_imagenet_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    return train_transform, test_transform


def get_CIFAR_transforms():
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
        ]
    )

    # extra transfrom for the training data, in order to achieve better performance
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
        ]
    )
    return train_transform, test_transform


if __name__ == "__main__":
    train: Dataset
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train, test = get_ImageNet_dataset(test_transform, test_transform)
    print(len(train))
    # get first 1000 samples from test_dataset as tensor of shape (1000, 3, 32, 32)
