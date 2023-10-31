import time

import torch
import torch.cuda as cuda
import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)
from tqdm import tqdm

from data_loader import get_balanced_dataloader
from src.analyzer import compute_X_reduced


def get_name_to_module(model):
    name_to_module = {}
    for m in model.named_modules():
        name_to_module[m[0]] = m[1]
    return name_to_module


def get_activation(all_outputs, name):
    def hook(model, input, output):
        all_outputs[name] = output.detach()

    return hook


def add_hooks(model, outputs, output_layer_names):
    """
    :param model:
    :param outputs: Outputs from layers specified in `output_layer_names` will be stored in `output` variable
    :param output_layer_names:
    :return:
    """
    name_to_module = get_name_to_module(model)
    for output_layer_name in output_layer_names:
        name_to_module[output_layer_name].register_forward_hook(
            get_activation(outputs, output_layer_name)
        )


class ModelWrapper(nn.Module):
    def __init__(self, model, output_layer_names, return_single=False):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_layer_names = output_layer_names
        self.outputs = {}
        self.return_single = return_single
        add_hooks(self.model, self.outputs, self.output_layer_names)

    def forward(self, images):
        self.model(images)
        output_vals = [
            self.outputs[output_layer_name] for output_layer_name in self.output_layer_names
        ]
        if self.return_single:
            return output_vals[0]
        else:
            return output_vals


def get_resnet50_feature_extractor(layer_name):
    core_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    output_layer_names = [layer_name]
    wrapper = ModelWrapper(core_model, output_layer_names, True)
    return wrapper


def train_layer_resnet50(layer_name, lr, wd, use_random_projection=False, optimizer_type="AdamW"):
    d = 8192
    # device = "cpu"
    device = "cuda"
    model = get_resnet50_feature_extractor(layer_name)
    # model = nn.DataParallel(model)
    model.to(device)
    # freeze model parameters
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dummy_data = torch.ones((1, 3, 224, 224)).to(device)
    dummy_output = model(dummy_data)
    dummy_output = dummy_output.reshape(dummy_output.shape[0], -1)
    if use_random_projection:
        dummy_output = dummy_output.detach().cpu().T
        dummy_output = compute_X_reduced(dummy_output, d)
        print(dummy_output.shape)
        dummy_output = dummy_output.to(device)

    epochs = 30
    classifer = nn.Linear(dummy_output.shape[1], 365)
    classifer.weight.data.normal_(mean=0.0, std=0.01)
    classifer.bias.data.zero_()

    # classifer = nn.DataParallel(classifer)
    classifer.to(device)
    data_name = "places"
    batch_size = 128
    print("loading data")
    train_dataloader, test_dataloader = get_balanced_dataloader(
        data_name, 100, 100, batch_size=batch_size
    )

    if optimizer_type == "AdamW":
        print("using AdamW")
        optimizer = torch.optim.AdamW(classifer.parameters(), lr=lr, weight_decay=wd)
    else:
        print("using SGD")
        optimizer = torch.optim.SGD(classifer.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    num_cycles = 4
    step_size_up = (len(train_dataloader) * 30) / (2 * num_cycles)
    # scheduler = CyclicLR(
    #     optimizer,
    #     base_lr=lr,
    #     max_lr=lr * 10,
    #     step_size_up=step_size_up,
    #     cycle_momentum=False,
    # )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total = 0
        correct = 0
        running_loss = 0
        for data, labels in tqdm(train_dataloader):
            data, labels = data.to(device), labels.to(device)
            features = model(data)
            features = features.reshape(features.shape[0], -1)
            if use_random_projection:
                features = features.detach().cpu().T
                features = compute_X_reduced(features, d)
                features = features.to(device)

            outputs = classifer(features)

            total += len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        print(f"epoch {epoch} finished acc {round(correct/total, 4)} loss {running_loss / total}")
    # if not use_random_projection:
    #     torch.save(classifer.state_dict(), f"resnet_50_{layer_name}_classifer.pth")
    # else:
    #     torch.save(classifer.state_dict(), f"resnet_50_{layer_name}_random_classifer.pth")
    # get accuracy
    classifer.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, labels in tqdm(test_dataloader):
            data, labels = data.to(device), labels.to(device)
            features = model(data)
            features = features.reshape(features.shape[0], -1)
            if use_random_projection:
                features = features.detach().cpu().T
                features = compute_X_reduced(features, d)
                features = features.to(device)
            outputs = classifer(features)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)
    acc = correct / total
    print(f"test acc: {acc} with opt : {optimizer_type} lr {lr} and wd {wd}")
    if not use_random_projection:
        torch.save(acc, f"acc/resnet_50_{layer_name}_{optimizer_type}_{lr}_{wd}_acc.pt")
    else:
        torch.save(acc, f"resnet_50_{layer_name}_random_acc.pt")


if __name__ == "__main__":
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    layer_name = "layer1.0.conv3"
    lrs = [1e-5, 1e-4, 1e-3]
    # lrs = [1e-3, 1e-2]

    wds = [1e-2, 1e-3]
    optimizers = ["SGD", "AdamW"]
    # wds = [1e-1]
    data_name = "places"
    batch_size = 128
    random_projection = False
    lr = 1e-3
    wd = 1e-2
    optimizer = "AdamW"
    train_layer_resnet50(layer_name, lr, wd, random_projection, optimizer)
    # for lr in lrs:
    #     for wd in wds:
    #         for optimizer in optimizers:
    # print(f"optimizer: {optimizer} lr: {lr} wd: {wd}")
    # start = time.time()

    # train_layer_resnet50(layer_name, lr, wd, random_projection, optimizer)
    # end = time.time()
    # print(f"total time: {end - start}")
    # acc = torch.load(f"acc/resnet_50_{layer_name}_{optimizer}_circle_{lr}_{wd}_acc.pt")
    # print(f"optimizer: {optimizer} lr: {lr} wd: {wd} acc: {acc}")

    # model = get_resnet50_feature_extractor()
    # # model = nn.DataParallel(model)
    # model.to("cuda")
    # # freeze model parameters
    # model.eval()
    # for param in model.parameters():
    #     param.requires_grad = False
    # classifer = nn.Linear(802816, 365)
    # classifer.load_state_dict(torch.load("resnet_50_classifer.pth"))
    # classifer.to("cuda")
    # classifer.eval()
    # total = 0
    # correct = 0
    # data_name = "places"
    # batch_size = 128
    # train_dataloader, test_dataloader = get_balanced_dataloader(
    #     data_name, 100, 100, batch_size=batch_size
    # )
    # with torch.no_grad():
    #     for data, labels in tqdm(test_dataloader):
    #         data, labels = data.to("cuda"), labels.to("cuda")
    #         features = model(data)
    #         features = features.reshape(features.shape[0], -1)
    #         outputs = classifer(features)
    #         correct += (outputs.argmax(1) == labels).sum().item()
    #         total += len(labels)
    # print(correct)
    # acc = correct / total
    # print(acc)
    # torch.save(acc, "resnet_50_acc.pth")

    # model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # data_name = "places"
    # batch_size = 128
    # model.fc = nn.Linear(512, 365)
    # model.to("cuda")
    # train_dataloader, test_dataloader = get_balanced_dataloader(
    #     data_name, 100, 100, batch_size=batch_size
    # )
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    # criterion = nn.CrossEntropyLoss()
    # epochs = 30
    # for epoch in range(epochs):
    #     total = 0
    #     correct = 0
    #     running_loss = 0
    #     for data, labels in tqdm(train_dataloader):
    #         data, labels = data.to("cuda"), labels.to("cuda")
    #         outputs = model(data)
    #         total += len(labels)
    #         correct += (outputs.argmax(1) == labels).sum().item()

    #         optimizer.zero_grad()
    #         loss = criterion(outputs, labels)
    #         running_loss += loss.item()
    #         loss.backward()
    #         optimizer.step()
    #     print(f"epoch {epoch} finished acc {round(correct/total, 4)} loss {running_loss / total}")

    # with torch.no_grad():
    #     total = 0
    #     correct = 0
    #     for data, labels in tqdm(test_dataloader):
    #         data, labels = data.to("cuda"), labels.to("cuda")
    #         outputs = model(data)
    #         total += len(labels)
    #         correct += (outputs.argmax(1) == labels).sum().item()
    # acc = correct / total
    # print(f"test acc: {acc}")
    # torch.save(acc, "acc/resnet_50_full_acc.pt")
