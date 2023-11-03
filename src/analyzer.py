import gc
import os
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from flowtorch.analysis import SVD
from torch import nn, optim
from torchvision.models.swin_transformer import SwinTransformerBlock
from tqdm import tqdm

from models.models import IterativeKNN
from utils.utils import (
    compute_X_reduced,
    get_size,
    mean_center,
    random_projection_method,
    vectorize_global_avg_pooling,
    vectorize_global_max_pooling,
)


def random_projection_method(X, b, cov=False):
    X_reduced = compute_X_reduced(X, b).to("cuda")

    variance = torch.var(X_reduced)

    if not cov:
        s = torch.linalg.svdvals(X_reduced)
        del X_reduced

        return s.detach().cpu(), variance.detach().cpu()
    else:
        cov_mat = torch.cov(X_reduced, correction=1)
        print(f"cov_mat shape: {cov_mat.size()}")
        del X_reduced
        s = torch.linalg.svdvals(cov_mat)
        return s.detach().cpu(), variance.detach().cpu()


class Analyzer(metaclass=ABCMeta):
    def __init__(self, model, name, data_name):
        self.name = name
        self.data_name = data_name
        self.cov = False
        # freeze all layers in the model
        for param in model.parameters():
            param.requires_grad = False
        self.model = model

        self.init_variables()

    def init_variables(self):
        if "cifar" in self.data_name:
            self.dummy_data = torch.randn(1, 3, 32, 32)
            self.b = 8192
        else:
            self.dummy_data = torch.randn(1, 3, 224, 224)
            self.b = 13000

        if self.data_name in ["cifar10", "imagenet"]:
            self.OOD = False
        elif self.data_name in ["cifar100", "places"]:
            self.OOD = True

        self.singular_values = []
        self.variances = []
        self.ranks = []
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

        # if self.dummy_data.size(-1) > 32:
        #     self.random_projection_train = True
        self.random_projection_train = False

    def init_G(self, sample_size):
        G = torch.randn(self.b, sample_size)
        self.G = G / torch.norm(G, dim=0, keepdim=True)

    @abstractmethod
    def get_layers(self):
        pass

    # register any hooks
    def register_hooks(self, func: callable):
        for layer in self.layers:
            hook = layer.register_forward_hook(func)
            self.hooks.append(hook)

    # save target layer
    def hook_save_target_layer(self, module, input, output):
        if self.layer_num == self.target:
            output = self.preprocess_output(output)
            torch.save(output, f"values/representations/{self.name}/{self.target}.pt")
        self.layer_num += 1

    # attach full layer
    def hook_full(self, module, input, output):
        output = self.preprocess_output(output)
        self.representations.append(output)

    def hook_vectorization(self, patch_size=2):
        if self.data_name in ["places", "imagenet"]:
            patch_size = 4
        print(f"patch size: {patch_size}")

        def hook_vectorization_helper(module, input, output):
            output = vectorize_global_avg_pooling(output, patch_size)
            self.representations.append(output)

        return hook_vectorization_helper

    def hook_save_representation(self, module, input, output):
        output = self.preprocess_output(output)
        target = self.labels

        torch.save(
            output, f"values/{self.data_name}/representations/{self.name}_{self.feature_num}.pt"
        )
        self.feature_num += 1

    # singular values
    def hook_compute_singular_values(self, module, input, output):
        output = self.preprocess_output(output).T
        if output.size(1) == 1:
            return
        print(f"output shape: {output.size()}")
        d = output.size(0)
        N = output.size(1)
        if d <= N:
            print(f"use full matrix : {output.size()}")
            output.to("cuda")
            cov_mat = torch.cov(output, correction=1)
            variance = torch.var(output)
            singular_values = torch.linalg.svdvals(cov_mat).detach().cpu()
            del cov_mat
        else:
            print(f"use random projection method : {output.size()}")
            singular_values, variance = random_projection_method(output, self.b, self.cov)

        print(singular_values[:10])
        del input
        del output
        self.variances.append(variance)
        self.singular_values.append(singular_values)

    def hook_compute_correlation_singular_values(self, module, input, output):
        output = self.preprocess_output(output).T
        if output.size(1) == 1:
            return
        print(f"output shape: {output.size()}")
        d = output.size(0)
        N = output.size(1)
        if d <= N:
            print(f"use full matrix : {output.size()}")
            output.to("cuda")
            cov_mat = torch.cov(output, correction=1)
            variance = torch.var(output)
            singular_values = torch.linalg.svdvals(cov_mat).detach().cpu()
            del cov_mat
        else:
            print(f"use random projection method : {output.size()}")
            singular_values, variance = random_projection_method(output, self.b, self.cov)

        print(singular_values[:10])
        del input
        del output
        self.variances.append(variance)
        self.singular_values.append(singular_values)

    def hook_store_random_projected_embeddings(self, module, input, output):
        output = self.preprocess_output(output).detach().cpu().T
        if output.size(1) == 1:
            return

        if output.size(0) <= self.b:
            output = torch.mm(self.G, output)
        self.representations.append(output)

    def hook_flowtorch_rank(self, module, input, output):
        output = self.preprocess_output(output).detach().cpu().T
        if output.size(1) == 1:
            return
        print(f"use random projection method : {output.size()}")
        X_reduced = compute_X_reduced(output, self.b).to("cuda")
        rank = SVD(X_reduced).rank
        print(f"rank: {rank}")
        self.ranks.append(rank)

    def save_flowtorch_rank(self, input_data):
        self.register_hooks(self.hook_flowtorch_rank)
        self.forward(input_data)

        save_path = f"values/{self.data_name}/rank"
        self.check_folder(save_path)

        torch.save(self.ranks, f"{save_path}/{self.name}_flowrank.pt")
        self.remove_hooks()

    def save_dimensions(self):
        # self.register_full_hooks()
        self.register_hooks(self.hook_full)
        self.forward(self.dummy_data)
        dims = []
        for layer in self.representations:
            print(layer.size(1))
            dims.append(layer.size(1))
        name = self.name.split("_")[0]

        # check is the folder exist
        path = f"values/{self.data_name}/dimensions"
        self.check_folder(path)

        torch.save(dims, f"{path}/{name}.pt")

        self.remove_hooks()

    # save target layer
    def save_target_layer(self, input_data):
        # self.register_save_target_layer()
        self.register_hooks(self.hook_save_target_layer)
        self.forward(input_data)
        self.remove_hooks()

    def inspect_layers_dim(self, sample_size=15000):
        # self.register_full_hooks()
        self.register_hooks(self.hook_full)
        self.forward(self.dummy_data.to(self.main_device))
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
        self.forward(self.dummy_data.to(self.main_device))
        if "cifar" in self.data_name:
            num_classes = 10
        else:
            num_classes = 1000

        for i, representation in enumerate(self.representations):
            cur_classifier = nn.Linear(representation.size(1), num_classes)

            cur_optim = optim.Adam(cur_classifier.parameters(), lr=0.001)
            cur_classifier.to(self.classifier_device)
            self.classifiers.append(cur_classifier)
            self.optimizers.append(cur_optim)

    def init_knns(self):
        self.knns = []
        for i, layer in enumerate(self.layers):
            cur_knn = IterativeKNN(n_neighbors=5)
            self.knns.append(cur_knn)

    def check_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    @abstractmethod
    def preprocess_output(self, output) -> torch.FloatTensor:
        pass

    def forward(self, input):
        # reset representations
        self.singular_values = []
        self.representations = []
        _ = self.model(input)

    def download_singular_values(self, input_data):
        # self.register_singular_hooks()
        self.register_hooks(self.hook_compute_singular_values)
        self.forward(input_data)
        singular_save_path = f"values/{self.data_name}/singular_values_direct"
        variance_path = f"values/{self.data_name}/variances"
        self.check_folder(singular_save_path)
        self.check_folder(variance_path)

        # check is the folder exist
        torch.save(self.singular_values, f"{singular_save_path}/{self.name}.pt")
        torch.save(self.variances, f"{variance_path}/{self.name}.pt")
        self.remove_hooks()

    def hook_compute_cov_variances(self, module, input, output):
        output = self.preprocess_output(output).T
        if output.size(1) == 1:
            return
        print(f"output shape: {output.size()}")
        d = output.size(0)
        N = output.size(1)
        if d <= N:
            print(f"use full matrix : {output.size()}")
            output.to("cuda")
            cov_mat = torch.cov(output, correction=1)
            variance = torch.var(cov_mat).detach().cpu()
        else:
            print(f"use random projection method : {output.size()}")
            reduced_X = compute_X_reduced(output, self.b).to("cuda")
            cov_mat = torch.cov(reduced_X, correction=1)
            print(cov_mat.shape)
            print(cov_mat)
            variance = torch.var(cov_mat).detach().cpu()
        print(variance)

        del input
        del output
        self.variances.append(variance)

    def register_cov_variances_hooks(self):
        for layer in self.layers:
            hook = layer.register_forward_hook(self.hook_compute_cov_variances)
            self.hooks.append(hook)

    def add_gpus(self, main_device, classifier_device):
        self.main_device = main_device
        self.classifier_device = classifier_device

    def download_cov_variances(self, input_data):
        self.register_cov_variances_hooks()
        self.forward(input_data)
        torch.save(self.variances, f"values/{self.data_name}/variances/{self.name}_cov.pt")
        self.remove_hooks()

    def download_knn_accuracy(
        self, train_data_loader, test_dataloader, OOD, feature_type="original", normalize=False
    ):
        if feature_type != "original":
            print(f"feature type: {feature_type}")
            self.register_hooks(self.hook_vectorization(feature_type, normalize))
        else:
            self.register_hooks(self.hook_full)
        self.init_knns()
        for data, target in tqdm(train_data_loader):
            self.forward(data)
            for representation, knn in zip(self.representations, self.knns):
                knn.update(representation, target)

        for knn in self.knns:
            knn.train()
        accuracies = []

        for data, target in test_dataloader:
            self.forward(data)
            for representation, knn in zip(self.representations, self.knns):
                knn.predict(representation, target)

        for knn in self.knns:
            accuracies.append(knn.get_accuracy())

        print(f"accuracy: {accuracies}")
        save_dir = f"values/{'cifar10' if 'cifar' in self.data_name else 'imagenet'}/{'knn_acc' if not self.OOD else 'knn_ood_acc'}/"
        self.check_folder(save_dir)
        torch.save(accuracies, f"{save_dir}/{self.name}_{'norm' if normalize else ''}.pt")

    def download_accuarcy(self, train_dataloader, test_dataloader):
        # create folder
        save_path = f"values/{'cifar10' if 'cifar' in self.data_name else 'imagenet'}/{'acc' if not self.OOD else f'ood_acc/{self.data_name}'}/{self.name}"
        self.check_folder(save_path)

        # register hooks
        self.register_hooks(self.hook_vectorization())

        # init classifiers
        self.init_classifers()
        self.train_classifers(train_dataloader)
        accuracies = self.test_classifers(test_dataloader)

        i = 0
        file_path = f"{save_path}/{i}.pt"
        while os.path.exists(file_path):
            i += 1
            file_path = f"{save_path}/{i}.pt"
        print(file_path)

        torch.save(accuracies, file_path)

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
                data, target = data.to(self.main_device), target.to(self.classifier_device)
                self.forward(data)
                for representation, classifier, optimizer in zip(
                    self.representations, self.classifiers, self.optimizers
                ):
                    optimizer.zero_grad()
                    representation = representation.to(self.classifier_device)
                    output = classifier(representation)
                    loss = self.criterion(output, target)
                    loss.backward(retain_graph=True)
                    optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}")

    def test_classifers(self, test_loader):
        self.model.eval()
        for classifier in self.classifiers:
            classifier.eval()
        accuracies = [0 for _ in range(len(self.classifiers))]
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.main_device), target.to(self.classifier_device)
                self.forward(data)

                for idx, (representation, classifier) in enumerate(
                    zip(self.representations, self.classifiers)
                ):
                    representation = representation.to(self.classifier_device)
                    output = classifier(representation)
                    # loss = self.criterion(output, target)
                    correct = output.argmax(dim=1).eq(target).sum().item()
                    accuracies[idx] += correct
        accuracies = [
            round(100.0 * correct / len(test_loader.dataset), 2) for correct in accuracies
        ]
        print(f"accuracy: {accuracies}")
        return accuracies
        # plot accuracy


class ResNetAnalyzer(Analyzer):
    def __init__(self, model, model_name, data_name):
        super().__init__(model, model_name, data_name)

    def get_layers(self):
        layers = []
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
                    layer_to_add = [layer.conv1, layer.conv2, layer.conv3]
                except:
                    layer_to_add = [layer.conv1, layer.conv2]
                layers.extend(layer_to_add)
        print(f"new layer counts {len(layers)}")
        # remove first layer
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
    def __init__(self, model, model_name, data_name):
        super().__init__(model, model_name, data_name)

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


class ConvNextV2Analyzer(Analyzer):
    def __init__(self, model, model_name, data_name):
        super().__init__(model, model_name, data_name)

    def get_layers(self):
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                layers.append(module)
        return layers

    def preprocess_output(self, output) -> torch.FloatTensor:
        output = output.reshape(output.size(0), -1)
        return output


class SwinAnalyzer(Analyzer):
    def __init__(self, model, model_name, data_name):
        super().__init__(model, model_name, data_name)

    def get_layers(self):
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, SwinTransformerBlock):
                layers.append(module)
        return layers

    def preprocess_output(self, output):
        output = output.view(output.size(0), -1)
        return output


class DINOV2Analyzer(Analyzer):
    def __init__(self, model, model_name, data_name):
        super().__init__(model, model_name, data_name)

    def get_layers(self):
        layers = []
        for name, module in self.model.named_modules():
            if "NestedTensorBlock" in str(
                type(module)
            ):  # checking the type of the module by its string representation
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
    elif "convnextv2" in model_name.lower():
        return ConvNextV2Analyzer(model, model_name, dummy_input)
    elif "convnext" in model_name.lower():
        return ConvNextAnalyzer(model, model_name, dummy_input)
    elif "swin" in model_name.lower():
        print(f"loading {model_name} analyzer")
        return SwinAnalyzer(model, model_name, dummy_input)
    elif "dinov2" in model_name.lower():
        print(f"loading {model_name} analyzer")
        return DINOV2Analyzer(model, model_name, dummy_input)
    else:
        raise ValueError("model name not supported")
