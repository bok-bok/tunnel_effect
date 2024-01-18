import argparse
import logging
import os

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_loader import (
    get_CIFAR_data_loader,
    get_data_loader,
    get_ImageNet100_dataloader,
)
from models.models import get_resnet18_by_resolution


def train(model, device, train_dataset, test_dataset, learning_rate, weight_decay, epochs):
    save_dir = f"weights/resnet18_resolution_aug/{image_resolution}/{str(learning_rate)}/{str(weight_decay)}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_name = f"{image_resolution}"

    logging.basicConfig(
        filename=f"{save_dir}/training_log.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
    )
    # scaler = GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1)

    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        for data, target in train_dataset:
            data, target = data.to(device), target.to(device)

            # with autocast():
            #     output = model(data)
            #     loss = criterion(output, target)

            # optimizer.zero_grad()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += output.argmax(dim=1).eq(target).sum().item()
            running_loss += loss.item() * data.size(0)

        running_loss = running_loss / len(train_dataset.dataset)
        train_acc = round(100.0 * correct / len(train_dataset.dataset), 2)
        val_loss, test_acc = validate(model, test_dataset, criterion, device)
        logging.info(
            f"epoch: {epoch} | loss: {running_loss:.2f} |val_L :{val_loss:.2f} | acc: {train_acc}% | test_acc: {test_acc}%"
        )
        print(
            f"epoch: {epoch} | loss: {running_loss:.2f} |val_L :{val_loss:.2f} | acc: {train_acc}% | test_acc: {test_acc}%"
        )
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{save_dir}/{model_name}_{epoch}_{test_acc}.pth")

        # check early stop
        # if early_stopper.early_stop(val_loss):
        #     print("Early stopping")
        #     break

        if scheduler is not None:
            scheduler.step()
    torch.save(model.state_dict(), f"{save_dir}/{model_name}_{epoch}_{test_acc}.pth")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True, help="Model name")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")

    parser.add_argument("--lr", type=float, required=True, help="Learning Rate")
    parser.add_argument("--wd", type=float, required=True, help="Weight Decay")
    parser.add_argument("--epochs", type=int, required=True, help="Epochs")

    parser.add_argument("--gpu", type=int, required=True, help="First GPU number to use")

    args = parser.parse_args()
    return (
        args.size,
        args.batch_size,
        args.lr,
        args.wd,
        args.epochs,
        args.gpu,
    )


if __name__ == "__main__":
    image_resolution, batch_size, learning_rate, weight_decay, epochs, device = parser()
    class_num = 100
    residual_connection = True
    augmentation = True

    device = f"cuda:{device}"
    train_dataloader, test_dataloader = get_ImageNet100_dataloader(
        resolution_size=image_resolution,
        batch_size=batch_size,
        classes_num=100,
        use_all=True,
        augmentation=augmentation,
    )

    # train_dataloader, test_dataloader = get_CIFAR_data_loader(augmentation=True)

    model = get_resnet18_by_resolution(
        image_resolution,
        class_num=class_num,
        residual_connection=residual_connection,
        pretrained=False,
    )
    model.train()

    train(
        model,
        device,
        train_dataloader,
        test_dataloader,
        learning_rate,
        weight_decay,
        epochs,
    )
