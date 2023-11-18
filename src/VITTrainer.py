import os
import argparse

from tqdm import tqdm


import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler

from torchvision import datasets, transforms
from transformers import ViTConfig,ViTModel, ViTForImageClassification

from data_loader import get_data_loader, get_ImageNet100_dataloader
# from utils.utils import EarlyStopper

IMAGENET_100_DIR = "/home/tolga/data/imagenet100"



def train(args, model, train_dataloader, test_dataloader):
    
    device = args.gpu
    learning_rate = args.lr
    weight_decay = args.wd
    epochs = args.epochs
    
    save_dir = f"weights/ViT/vit_{args.img_size}/{str(learning_rate) + '_' + str(weight_decay)}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_name = f"{args.img_size}"

    # scaler = GradScaler()
    # early_stopper = EarlyStopper(patience=10)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1)

    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        for data, target in train_dataloader:
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

        running_loss = running_loss / len(train_dataloader.dataset)
        train_acc = round(100.0 * correct / len(train_dataloader.dataset), 2)
        val_loss, test_acc = validate(model, test_dataloader, criterion, device)
        print(
            f"epoch: {epoch} | loss: {running_loss:.2f} |val_L :{val_loss:.2f} | acc: {train_acc}% | test_acc: {test_acc}%"
        )
        if epoch % 10 == 0:
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


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ViT Trainer')
    
    parser.add_argument("--img_size", type=int, default = 224)
    parser.add_argument("--patch_size", type=int,default = 8)
    parser.add_argument("--batch_size", type=int, default = 256)
    parser.add_argument("--lr", type=float, default = 0.1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num_labels", type=int, default = 100)
    parser.add_argument("--wd", type=float, default = 0.001)
    parser.add_argument("--gpu", type=int, default = 0)
    
    args = parser.parse_args()
        
    train_dataloader, test_dataloader = get_ImageNet100_dataloader(resolution_size = args.img_size,
                                                                   batch_size = args.batch_size,
                                                                   classes_num = args.num_labels,
                                                                   use_all = False)
    
    config = ViTConfig(image_size = args.img_size, 
                       patch_size = args.patch_size, 
                       num_labels = args.num_labels)
    
    vit_model = ViTForImageClassification(config)
    
    train(args,vit_model,train_dataloader,test_dataloader)


    
    # trainer = pl.Trainer(accelerator='auto',devices='auto')
    # trainer.fit(vit_model,train_dataloaders=train_dataloader,val_dataloaders=test_dataloader)
    
    
    
    