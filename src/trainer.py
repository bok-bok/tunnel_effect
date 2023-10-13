import torch
from torch import optim
from tqdm import tqdm

from data_loader import get_data_loader
from models.models import MLP


def train(model, train_dataset, test_dataset, criterion, optimizer, epochs):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model.train()
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        for data, target in tqdm(train_dataset):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            correct += output.argmax(dim=1).eq(target).sum().item()
            loss = criterion(output, target)
            loss.backward()

            running_loss += loss.item() * data.size(0)
            optimizer.step()

        # print accuracy and loss for a epoch
        acc = round(100.0 * correct / len(train_dataset.dataset), 2)
        print(f"Epoch {epoch+1}/{epochs} | loss: {running_loss:.4f} | acc: {acc}%")
        validate(model, test_dataset, criterion, device)
        # save every 10th epoch
    torch.save(model.state_dict(), f"weights/mlp_new.pth")


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
    print(f"validation ----- loss: {running_loss:.4f} | acc: {acc}%")


if __name__ == "__main__":
    model = MLP()
    train_dataloader, test_dataloader = get_data_loader("cifar10", batch_size=128)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.0, weight_decay=0)

    train(model, train_dataloader, test_dataloader, torch.nn.CrossEntropyLoss(), optimizer, 1000)
