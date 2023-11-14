
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from timm.models.vision_transformer import VisionTransformer

class ViTTrainer:
    def __init__(self, dataset_name, img_size, patch_size, in_chans, num_classes, batch_size=32, num_workers=4, learning_rate=0.001, num_epochs=10):
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if self.dataset_name == 'imagenet':
            self.train_dataset = datasets.ImageNet(root='./data', split='train', transform=self.transform)
            self.val_dataset = datasets.ImageNet(root='./data', split='val', transform=self.transform)
        elif self.dataset_name == 'cifar10':
            self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
            self.val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
        else:
            raise ValueError('Invalid dataset name')

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        self.model = VisionTransformer(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, num_classes=self.num_classes)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

            self.validate()

    def validate(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.val_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the %d test images: %d %%' % (len(self.val_dataset), 100 * correct / total))
