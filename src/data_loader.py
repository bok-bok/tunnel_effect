import os

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
from torchvision import datasets, transforms

PLACE_DIR = "/data/datasets/Places/places365_standard/small_easyformat"
IMAGENET_DIR = "/data/datasets/ImageNet1K"
IMAGENET_DIR = "/data/datasets/ImageNet2012"
DIR_DICT = {"places": PLACE_DIR, "imagenet": IMAGENET_DIR}


def get_data_loader(dataset_name, batch_size=512):
    if "cifar" in dataset_name:
        train_transform, test_transform = get_CIFAR_transforms()
        if "100" in dataset_name:
            return get_CIFAR_100_data_loader(train_transform, test_transform, batch_size)
        elif "10" in dataset_name:
            return get_CIFAR_data_loader(train_transform, test_transform, batch_size)
    elif dataset_name == "imagenet" or dataset_name == "places":
        if dataset_name == "places":
            train_samples_per_class = 100
            test_samples_per_class = 100
        else:
            train_samples_per_class = 200
            test_samples_per_class = 50
        data_loader = get_balanced_dataloader(
            dataset_name, train_samples_per_class, test_samples_per_class, batch_size
        )
        return data_loader

    else:
        raise ValueError("dataset name not supported")


def get_CIFAR_100_data_loader(train_transform, test_transform, batch_size=512, use_previous=False):
    # get 10 random classes from CIFAR100
    train_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, transform=train_transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, transform=test_transform, download=True
    )

    print("loading CIFAR100 data")
    if use_previous:
        print("loading previous indices")
        train_indices = torch.load("values/cifar10/indices/train_indices.pt")
        test_indices = torch.load("values/cifar10/indices/test_indices.pt")
        classes = torch.load("values/cifar10/indices/classes.pt")
    else:
        print("creating new indices")
        classes = torch.randperm(100)[:10].tolist()
        # Get indices of desired classes
        train_indices = [
            i for i in range(len(train_dataset)) if train_dataset.targets[i] in classes
        ]
        test_indices = [i for i in range(len(test_dataset)) if test_dataset.targets[i] in classes]
        # torch.save(train_indices, "values/cifar10/indices/train_indices.pt")
        # torch.save(test_indices, "values/cifar10/indices/test_indices.pt")
        # torch.save(classes, "values/cifar10/indices/classes.pt")

    class_mapping = {original: new for new, original in enumerate(classes)}
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


# dataloader for places and imagenet
def get_balanced_dataloader(
    data_name, train_samples_per_class, test_samples_per_class, batch_size=512, classes=None
):
    train_dataset, test_dataset = get_balanced_dataset(
        data_name, train_samples_per_class, test_samples_per_class, classes
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_dataloader, test_dataloader


# for places and imagenet
def get_balanced_dataset(data_name, train_samples_per_class, test_samples_per_class, classes=None):
    dataset_dir = DIR_DICT[data_name]
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    train_transform, test_transform = get_imagenet_transforms()

    # create dataset
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=test_transform)

    # get indices
    train_indices = get_balanced_indices(
        train_dataset, data_name, "train", train_samples_per_class, classes
    )
    test_indices = get_balanced_indices(
        val_dataset, data_name, "val", test_samples_per_class, classes
    )
    print(f"train indices: {len(train_indices)}")
    print(f"test indices: {len(test_indices)}")

    # create subset with indices
    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(val_dataset, test_indices)
    return train_dataset, test_dataset


def get_balanced_imagenet_indices(dataset, dataset_type, sample_size=15000):
    if dataset_type not in ["train", "val"]:
        raise ValueError("dataset_type must be train or val")
    num_classes = len(dataset.classes)
    samples_per_class = sample_size // num_classes

    indices = []

    for class_idx in range(num_classes):
        class_indices = np.where(np.array(dataset.targets) == class_idx)[0]
        class_sample_indices = np.random.choice(class_indices, samples_per_class, replace=False)

        indices.extend(class_sample_indices)
    torch.save(indices, f"values/{dataset_type}_balanced_indices.pt")
    return indices


def get_balanced_indices(dataset, dataset_name, dataset_type, samples_per_class=100, classes=None):
    # create dir if not exits
    save_dir = f"values/{dataset_name}/indices"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_classes = len(dataset.classes)
    if classes is not None:
        num_classes = classes

    # if file exists, return
    file_dir = os.path.join(save_dir, f"{dataset_type}_{num_classes}_{samples_per_class}.pt")
    if os.path.exists(file_dir):
        print(f"loading saved {dataset_name} balanced indices")
        return torch.load(file_dir)

    if dataset_type not in ["train", "val"]:
        raise ValueError("dataset_type must be train or val")

    # create new file if not exists
    # get indices
    print("creating new indices")
    indices = []
    for class_idx in range(num_classes):
        class_indices = np.where(np.array(dataset.targets) == class_idx)[0]
        class_sample_indices = np.random.choice(class_indices, samples_per_class, replace=False)

        indices.extend(class_sample_indices)
    save_path = f"{save_dir}/{dataset_type}_{num_classes}_{samples_per_class}.pt"
    torch.save(indices, save_path)
    return indices


def get_balanced_imagenet_input_data(sample_size=15000, use_previous=True):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    val_dir = os.path.join(cur_dir, "/data/datasets/ImageNet1K/val")
    _, test_transform = get_imagenet_transforms()
    val_dataset = datasets.ImageFolder(root=val_dir, transform=test_transform)
    # I want to get sample_size samples from val_dataset but with classes balanced

    if not use_previous:
        indices = get_balanced_imagenet_indices(val_dataset, "val", sample_size)
    else:
        indices = torch.load("values/balanced_indices.pt")
    balanced_dataset = Subset(val_dataset, indices)
    balanced_dataloader = DataLoader(balanced_dataset, batch_size=sample_size, shuffle=False)
    input_data = next(iter(balanced_dataloader))[0]
    return input_data


def get_cifar_input_data(sample_size=10000):
    train_transform, test_trainsform = get_CIFAR_transforms()
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=test_trainsform, download=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=sample_size, shuffle=False)
    input_data = next(iter(test_dataloader))[0]
    return input_data


def get_imagenet_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    _, _ = get_data_loader("imagenet", batch_size=512)
    # _, _ = get_data_loader("places", batch_size=512)
