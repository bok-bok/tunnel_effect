import argparse
import os

import torch
from torch import nn, optim
from torchvision.models import vgg13
from tqdm import tqdm

from data_loader import get_data_loader, get_ImageNet100_dataloader
from models.models import MLP, get_resnet34_imagenet100


def train(model, device, train_dataset, test_dataset, optimizer, epochs, scheduler):
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        for data, target in train_dataset:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            correct += output.argmax(dim=1).eq(target).sum().item()
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)

            loss.backward()
            optimizer.step()
        running_loss = running_loss / len(train_dataset.dataset)
        train_acc = round(100.0 * correct / len(train_dataset.dataset), 2)
        _, acc = validate(model, test_dataset, criterion, device)
        print(f"epoch: {epoch} | loss: {running_loss:.4f} | acc: {train_acc}% | test_acc: {acc}%")
        if scheduler is not None:
            scheduler.step()
        # print accuracy and loss for a epoch
    # print(f"Train acc: {train_acc}%")
    # print(f"Test acc: {acc}%")


def validate(model, test_dataset, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataset:
            data, target = data.to(device), target.to(device)
            output = model(data)
            correct += output.argmax(dim=1).eq(target).sum().item()
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)

    # print accuracy and loss for a epoch
    acc = round(100.0 * correct / len(test_dataset.dataset), 2)
    running_loss = running_loss / len(test_dataset.dataset)
    # print(f"validation ----- loss: {running_loss:.4f} | acc: {acc}%")
    return running_loss, acc


def parser():
    parser = argparse.ArgumentParser(description="Get GPU numbers")
    parser.add_argument("--size", type=int, required=True, help="Model name")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")

    parser.add_argument("--lr", type=float, required=True, help="Batch size")
    parser.add_argument("--momentum", type=float, required=True, help="Batch size")
    parser.add_argument("--wd", type=float, required=True, help="Batch size")
    parser.add_argument("--epochs", type=int, required=True, help="Batch size")
    parser.add_argument("--milestones", type=int, nargs="+", required=True, help="Batch size")

    parser.add_argument("--gpu", type=int, required=True, help="First GPU number to use")

    args = parser.parse_args()
    return (
        args.size,
        args.batch_size,
        args.lr,
        args.momentum,
        args.wd,
        args.epochs,
        args.milestones,
        args.gpu,
    )


if __name__ == "__main__":
    size, batch_size, learning_rate, momentum, wd, epochs, milestones, device = parser()

    device = f"cuda:{device}"

    train_dataloader, test_dataloader = get_ImageNet100_dataloader(
        size, batch_size, classes_num=None, use_all=True
    )

    model = get_resnet34_imagenet100(size, pretrained=False)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    train(
        model,
        device,
        train_dataloader,
        test_dataloader,
        optimizer,
        epochs,
        scheduler,
    )
    save_dir = f"weights/resnet34_imagenet100/{size}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_name = f"{learning_rate}_{momentum}_{wd}_{epochs}.pth"
    torch.save(model.state_dict(), f"{save_dir}/{model_name}")
