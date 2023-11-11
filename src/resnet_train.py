import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from tqdm import tqdm

model = resnet34(pretrained=False)


model.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
model.fc = nn.Linear(512, 10)


train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
    ]
)
test_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)),
    ]
)

# extra transfrom for the training data, in order to achieve better performance

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, transform=train_transform, download=True
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=test_transform, download=True
)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)


learning_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
num_epochs = 164
mini_batch_size = 128
lr_decay_milestones = [82, 123]
lr_decay_gamma = 0.1
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=lr_decay_milestones, gamma=lr_decay_gamma
)
device = "cuda:1"
model.to(device)


for epoch in range(num_epochs):
    running_loss = 0.0
    correct_count = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(inputs)
        correct_count += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
    with torch.no_grad():
        val_correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_correct += output.argmax(dim=1).eq(target).sum().item()
    scheduler.step()
    print(
        f"Epoch: {epoch+1}, Loss: {running_loss/len(train_loader)} Accuracy: {round(correct_count/len(train_dataset),2)} Val_ACC: {round(val_correct / len(test_dataset),2)}"
    )


torch.save(model.state_dict(), f"resnet_true.pth")
