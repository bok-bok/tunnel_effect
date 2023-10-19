import time
from abc import ABCMeta, abstractmethod

import torch
from torch import nn, optim
from tqdm import tqdm


def mean_center(X):
    mean = torch.mean(X, dim=1, keepdim=True)
    X_prime = X - mean
    return X_prime


def random_projection_method(X, b, normalized=False):
    X = mean_center(X)
    G = torch.randn(b, X.size(0))
    G = G / torch.norm(G, dim=0, keepdim=True)  # Normalize columns to unit length
    X_reduced = torch.mm(G, X)
    S = torch.linalg.svdvals(X_reduced)
    S_squared = S**2
    if normalized:
        S_squared = S_squared / torch.sum(S_squared)
    return S_squared.detach().cpu()


class Analyzer(metaclass=ABCMeta):
    def __init__(self, model, name, dummy_input):
        self.dummy_data = dummy_input
        self.b = 8192
        self.singular_values = []
        self.skip_layers = False
        self.skip_numbers = 2
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.name = name
        self.total_samples = 0
        self.hooks = []
        for param in model.parameters():
            param.requires_grad = False

        self.model = model

        self.criterion = nn.CrossEntropyLoss()
        self.representations = []

        self.layers = self.get_layers()

    @abstractmethod
    def get_layers(self) -> list[nn.Module]:
        pass

    def register_full_hooks(self):
        for layer in self.layers:
            hook = layer.register_forward_hook(self.hook_full)
            self.hooks.append(hook)

    def register_singular_hooks(self):
        for layer in self.layers:
            hook = layer.register_forward_hook(self.hook_compute_singular_values)
            self.hooks.append(hook)

    def save_dimensions(self):
        self.register_full_hooks()
        self.forward(self.dummy_data)
        dims = []
        for layer in self.representations:
            dims.append(layer.size(1))
        name = self.name.split("_")[0]
        torch.save(dims, f"values/dimensions/{name}.pt")

        self.remove_hooks()

    def hook_full(self, module, input, output):
        output = self.preprocess_output(output)
        self.representations.append(output)

    def hook_compute_singular_values(self, module, input, output):
        output = self.preprocess_output(output).detach().cpu().T
        # skip dummy
        # output = output[:8000]
        # print(output.shape)
        # cov_mat = torch.cov(output, correction=1)
        # print(cov_mat.size())
        # singular_values = torch.linalg.matrix_rank(cov_mat)
        if output.size(1) == 1:
            return

        if output.size(0) <= self.b:
            print(f"use full matrix : {output.size()}")
            cov_mat = torch.cov(output, correction=1)
            singular_values = torch.linalg.svdvals(cov_mat)
        else:
            print(f"use random projection method : {output.size()}")
            singular_values = random_projection_method(output, self.b)
        print(singular_values[:10])
        self.singular_values.append(singular_values)

    def init_classifers(self):
        self.classifiers = []
        self.optimizers = []
        self.forward(self.dummy_data)

        for representation in self.representations:
            cur_classifier = nn.Linear(representation.size(1), 10)

            cur_optim = optim.Adam(cur_classifier.parameters(), lr=0.001)
            cur_classifier.to(self.device)
            self.classifiers.append(cur_classifier)
            self.optimizers.append(cur_optim)

    @abstractmethod
    def preprocess_output(self, output) -> torch.FloatTensor:
        pass

    def forward(self, input):
        # reset representations
        self.singular_values = []
        self.representations = []
        _ = self.model(input)

    def download_singular_values(self, input_data):
        self.register_singular_hooks()
        self.forward(input_data)
        torch.save(self.singular_values, f"values/singular_values/{self.name}_original_500.pt")
        self.remove_hooks()

    def download_accuarcy(self, train_data_loader, test_dataloader, OOD):
        self.register_full_hooks()
        self.init_classifers()
        self.train_classifers(train_data_loader)
        self.test_classifers(test_dataloader, OOD)
        self.remove_hooks()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def train_classifers(self, data_loader, epochs=30):
        # do not update model parameters
        # freeze all layers in the model
        self.model.eval()
        for layer in self.model.parameters():
            layer.requires_grad = False

        for epoch in range(epochs):
            for data, target in tqdm(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.forward(data)
                for representation, classifier, optimizer in zip(
                    self.representations, self.classifiers, self.optimizers
                ):
                    optimizer.zero_grad()
                    representation = representation.to(self.device)
                    output = classifier(representation)
                    loss = self.criterion(output, target)
                    loss.backward(retain_graph=True)
                    optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}")

    def test_classifers(self, test_loader, OOD):
        self.model.eval()
        for classifier in self.classifiers:
            classifier.eval()
        accuracies = [0 for _ in range(len(self.classifiers))]
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.forward(data)

                for idx, (representation, classifier) in enumerate(
                    zip(self.representations, self.classifiers)
                ):
                    representation = representation.to(self.device)
                    output = classifier(representation)
                    # loss = self.criterion(output, target)
                    correct = output.argmax(dim=1).eq(target).sum().item()
                    accuracies[idx] += correct
        accuracies = [
            round(100.0 * correct / len(test_loader.dataset), 2) for correct in accuracies
        ]
        print(f"accuracy: {accuracies}")
        if not OOD:
            torch.save(accuracies, f"values/acc/{self.name}.pt")
        else:
            torch.save(accuracies, f"values/ood_acc/{self.name}.pt")

        # plot accuracy


class ResNetAnalyzer(Analyzer):
    def __init__(self, model, model_name, dummy_input):
        super().__init__(model, model_name, dummy_input)

    def get_layers(self) -> list[nn.Module]:
        layers = []
        layer_count = len(list(i for i in self.model.named_modules() if isinstance(i, nn.Conv2d)))
        if layer_count > 40:
            print("skip odd layers")
            self.skip_layers = True
            count = 0
            for block_name in ["layer1", "layer2", "layer3", "layer4"]:
                block = getattr(self.model, block_name)
                for layer in block:
                    layers.extend([layer.conv1, layer.conv2])

            print(f"new layer count {len(layers)}")
        else:
            layers.append(self.model.resnet.conv1)
            for block_name in ["layer1", "layer2", "layer3", "layer4"]:
                block = getattr(self.model.resnet, block_name)
                for layer in block:
                    layers.extend([layer.conv1, layer.conv2])
            print(f"new layer count {len(layers)}")
        return layers

    def preprocess_output(self, output) -> torch.FloatTensor:
        # current input shape is (batch_size, channels, height, width) I want (channels * height * width, batch_size)
        output = output.view(output.size(0), -1)
        return output


class MLPAnalyzer(Analyzer):
    def __init__(self, model, model_name, dummy_input):
        super().__init__(model, model_name, dummy_input)

    def get_layers(self) -> list[nn.Module]:
        layers = []
        for layer in self.model.layers:
            if isinstance(layer, nn.Linear):
                layers.append(layer)
        return layers

    def preprocess_output(self, output):
        output = output.view(output.size(0), -1)
        return output
