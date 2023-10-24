import gc
import os
import time
from abc import ABCMeta, abstractmethod

import torch
from torch import nn, optim
from torchvision.models.swin_transformer import SwinTransformerBlock
from tqdm import tqdm

from utils.utils import get_size


def mean_center(X):
    mean = torch.mean(X, dim=1, keepdim=True)
    X_prime = X - mean
    return X_prime


def random_projection_method(X, b, normalized=False):
    X = mean_center(X)
    G = torch.randn(b, X.size(0))
    G = G / torch.norm(G, dim=0, keepdim=True)  # Normalize columns to unit length
    X_reduced = torch.mm(G, X)
    del G
    del X

    X_reduced = X_reduced.to("cuda")
    S = torch.linalg.svdvals(X_reduced)
    del X_reduced

    S_squared = S**2
    if normalized:
        S_squared = S_squared / torch.sum(S_squared)

    del S
    return S_squared.detach().cpu()


class Analyzer(metaclass=ABCMeta):
    def __init__(self, model, name, dummy_input):
        self.dummy_data = dummy_input
        self.name = name
        self.layer_num = 0
        self.target = 11

        # freeze all layers in the model
        for param in model.parameters():
            param.requires_grad = False
        self.model = model

        self.init_variables()

    def init_variables(self):
        self.b = 13000

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
        self.device = "cpu"
        self.total_samples = 0
        self.hooks = []
        self.representations = []

        self.criterion = nn.CrossEntropyLoss()

        self.layers = self.get_layers()

        if self.dummy_data.size(-1) > 32:
            self.random_projection_train = True

    def init_G(self, sample_size):
        G = torch.randn(self.b, sample_size)
        self.G = G / torch.norm(G, dim=0, keepdim=True)

    @abstractmethod
    def get_layers(self):
        pass

    def register_full_hooks(self):
        for layer in self.layers:
            hook = layer.register_forward_hook(self.hook_full)
            self.hooks.append(hook)

    def register_full_temp_hooks(self):
        for layer in self.layers:
            hook = layer.register_forward_hook(self.hook_save_target_layer)
            self.hooks.append(hook)

    def register_random_projection_hooks(self):
        for layer in self.layers:
            hook = layer.register_forward_hook(self.hook_store_random_projected_embeddings)
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

    def hook_save_target_layer(self, module, input, output):
        if self.layer_num == self.target:
            output = self.preprocess_output(output)
            torch.save(output, f"values/representations/{self.name}/{self.target}.pt")
        self.layer_num += 1

    def hook_full(self, module, input, output):
        output = self.preprocess_output(output)
        self.representations.append(output)

    def hook_compute_singular_values(self, module, input, output):
        output = self.preprocess_output(output).T
        if output.size(1) == 1:
            return
        print(f"output shape: {output.size()}")

        if output.size(0) <= self.b:
            print(f"use full matrix : {output.size()}")
            output.to("cuda")
            cov_mat = torch.cov(output, correction=1)
            singular_values = torch.linalg.svdvals(cov_mat).detach().cpu()
            del cov_mat

        else:
            print(f"use random projection method : {output.size()}")
            singular_values = random_projection_method(output, self.b)

        print(singular_values[:10])
        # torch.save(singular_values, f"values/singular_values/{self.name}/{self.layer_num}.pt")
        # self.layer_num += 1
        del input
        del output

        self.singular_values.append(singular_values)

    def hook_store_random_projected_embeddings(self, module, input, output):
        output = self.preprocess_output(output).detach().cpu().T
        if output.size(1) == 1:
            return

        if output.size(0) <= self.b:
            output = torch.mm(self.G, output)
        self.representations.append(output)

    def download_one_representation(self, input_data):
        # 11
        self.register_full_temp_hooks()
        self.forward(input_data)

    def inspect_layers_dim(self, sample_size=20000):
        self.register_full_hooks()
        self.forward(self.dummy_data)
        for representation in self.representations:
            dim = representation.view(representation.size(0), -1).size(1)
            representation_bytes = representation.element_size() * dim * sample_size
            g_bytes = representation.element_size() * dim * self.b
            reduced_bytes = representation.element_size() * self.b * sample_size
            total_bytes = representation_bytes + g_bytes + reduced_bytes
            size = get_size(total_bytes)
            print(f"layer dim: {dim}, size: {size} GB")

        self.remove_hooks()

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
        torch.save(self.singular_values, f"values/singular_values/{self.name}.pt")
        self.remove_hooks()

    def download_accuarcy(self, train_data_loader, test_dataloader, OOD):
        if self.random_projection_train:
            self.register_random_projection_hooks()
        else:
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

    def get_layers(self):
        layers = []
        layer_count = len(list(i for i in self.model.named_modules() if isinstance(i, nn.Conv2d)))
        if layer_count > 100:
            print("skip odd layers")
            self.skip_layers = True
            count = 0
            for block_name in ["layer1", "layer2", "layer3", "layer4"]:
                block = getattr(self.model, block_name)
                for layer in block:
                    layers.extend([layer.conv1, layer.conv2])

            print(f"new layer count {len(layers)}")
        else:
            # check if the model is modified
            try:
                # if resnet exist, then the model is modified
                self.model.resnet
                cur_model = self.model.resnet
            except:
                cur_model = self.model
            layers.append(cur_model.conv1)
            for block_name in ["layer1", "layer2", "layer3", "layer4"]:
                block = getattr(cur_model, block_name)
                for layer in block:
                    try:
                        layer.conv3
                        # layer_to_add = [layer.conv1, layer.conv2]
                        layer_to_add = [layer.conv1, layer.conv2, layer.conv3]
                    except:
                        layer_to_add = [layer.conv1, layer.conv2]
                    layers.extend(layer_to_add)
            print(f"new layer count {len(layers)}")

        return layers

    def preprocess_output(self, output) -> torch.FloatTensor:
        output = output.view(output.size(0), -1)
        return output


class MLPAnalyzer(Analyzer):
    def __init__(self, model, model_name, dummy_input):
        super().__init__(model, model_name, dummy_input)

    def get_layers(self):
        layers = []
        for layer in self.model.layers:
            if isinstance(layer, nn.Linear):
                layers.append(layer)
        return layers

    def preprocess_output(self, output):
        output = output.view(output.size(0), -1)
        return output


class ConvNextAnalyzer(Analyzer):
    def __init__(self, model, model_name, dummy_input):
        super().__init__(model, model_name, dummy_input)

    def extract_conv2d_from_cnblock(self, model, conv2d_layers):
        for name, module in model.named_children():
            if name.startswith("block"):
                for child_name, child_module in module.named_children():
                    if isinstance(child_module, nn.Conv2d):
                        conv2d_layers.append(child_module)
            else:
                # Recurse into other child modules
                self.extract_conv2d_from_cnblock(module, conv2d_layers)

    def get_layers(self):
        layers = []
        self.extract_conv2d_from_cnblock(self.model, layers)
        print(f"new layer count {len(layers)}")
        return layers

    def preprocess_output(self, output) -> torch.FloatTensor:
        output = output.reshape(output.size(0), -1)
        return output


class SwinAnalyzer(Analyzer):
    def __init__(self, model, model_name, dummy_input):
        super().__init__(model, model_name, dummy_input)

    def get_layers(self):
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, SwinTransformerBlock):
                layers.append(module)
        return layers

    def preprocess_output(self, output):
        output = output.view(output.size(0), -1)
        return output


def get_analyzer(model, model_name: str, dummy_input):
    if "mlp" in model_name:
        return MLPAnalyzer(model, model_name, dummy_input)

    elif "resnet" in model_name:
        return ResNetAnalyzer(model, model_name, dummy_input)
    elif "convnext" in model_name.lower():
        return ConvNextAnalyzer(model, model_name, dummy_input)
    elif "swin" in model_name.lower():
        print(f"loading {model_name} analyzer")
        return SwinAnalyzer(model, model_name, dummy_input)
    else:
        raise ValueError("model name not supported")
