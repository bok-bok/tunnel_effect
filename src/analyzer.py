import gc
import os
import sys
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances

sys.path.append("A-ViT")
# from timm.models.vision_transformer import Block
from timm.models.te_vision_transformer import Block
from torch import nn, optim
from torchvision.models.swin_transformer import SwinTransformerBlock
from torchvision.models.vision_transformer import EncoderBlock
from tqdm import tqdm

from models.models import IterativeKNN
from utils.utils import (
    average_pairwise_cosine_distance,
    compute_X_reduced,
    computeNC1,
    get_size,
    mean_center,
    random_projection_method,
    stable_rank,
    vectorize_avg_token,
    vectorize_global_avg_pooling,
    vectorize_global_avg_pooling_by_2,
    vectorize_global_max_pooling,
)


def random_projection_method(X, b, device):
    X_reduced = compute_X_reduced(X, b).to(device)
    print(f"X_reduced shape: {X_reduced.size()}")

    squared_s = torch.linalg.svdvals(X_reduced) ** 2
    del X_reduced

    return squared_s.detach().cpu()


class Analyzer(metaclass=ABCMeta):
    def __init__(self, model, name, data_name):
        self.name = name
        self.data_name = data_name
        self.cov = False
        # freeze all layers in the model
        for param in model.parameters():
            param.requires_grad = False
        self.model = model.to("cpu")

        self.init_variables()

    def init_variables(self):
        if "cifar" in self.data_name:
            self.dummy_data = torch.randn(1, 3, 32, 32)
            self.b = 8192
        else:
            self.dummy_data = torch.randn(1, 3, 224, 224, dtype=torch.float32)
            # self.b = 13000
            self.b = 8192

        if self.data_name in ["cifar10", "imagenet", "imagenet100"]:
            self.OOD = False
        elif self.data_name in ["cifar100", "places", "ninco"]:
            self.OOD = True
        self.lr = 0.001
        self.singular_values = []
        self.variances = []
        self.ranks = []
        self.skip_layers = False
        self.skip_numbers = 2
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.device = "cpu"
        self.total_samples = 0
        self.hooks = []
        self.representations = []

        self.criterion = nn.CrossEntropyLoss()

        self.layers = self.get_layers()
        print(f"layer counts: {len(self.layers)}")

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
    def register_hooks(self, func: callable, only_last_layer=False):
        # avit model does not call forward inside of block
        if "avit" in self.name.lower():
            for layer in self.layers:
                hook = layer.register_forward_act_hook(func)
                self.hooks.append(hook)
        else:
            if only_last_layer:
                # register only last layer
                hook = self.layers[-1].register_forward_hook(func)
                self.hooks.append(hook)
            else:
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
        output = output.detach().cpu().to(self.classifier_device)
        output = self.preprocess_output(output)

        self.representations.append(output)

    def hook_vectorization(self, resolution):
        print(f"resolution: {resolution}")
        print(f"data name: {self.data_name}")
        # if resolution == 224:
        #     patch_size = 6
        # elif resolution == 128:
        #     patch_size = 4
        # else:
        #     patch_size = 2
        if resolution == 224:
            patch_size = 4
        else:
            patch_size = 2

        def hook_vectorization_helper(module, input, output):
            output = output.detach().cpu().to(self.classifier_device)

            if len(output.size()) == 4:
                # print(f"patch size: {patch_size}")
                # # if cnn
                # output = vectorize_global_avg_pooling(
                #     output, patch_size, normalize=True, device=self.classifier_device
                # )
                output = vectorize_global_avg_pooling_by_2(output)
            elif len(output.size()) == 3:
                # if transformer
                output = vectorize_avg_token(output)

            self.representations.append(output)

        return hook_vectorization_helper

    def hook_compute_vectorized_singular_values(self, patch_size=2):
        if self.data_name in ["places", "imagenet", "imagenet100"]:
            patch_size = 4
        print(f"patch size: {patch_size}")

        def hook_compute_vectorzed_singular_values_helper(module, input, output):
            output = vectorize_global_avg_pooling(
                output, patch_size, normalize=False, device=self.main_device
            ).T

            print(f"output size : {output.size()}")
            # output = output.to(self.classifier_device)
            singular_values = random_projection_method(output, self.b, self.classifier_device)

            singular_values = torch.linalg.svdvals(output).detach().cpu()
            squared_s = singular_values**2
            print(squared_s[:10])
            self.singular_values.append(squared_s)

        return hook_compute_vectorzed_singular_values_helper

    def hook_save_representation(self, module, input, output):
        output = self.preprocess_output(output)
        target = self.labels

        torch.save(output, f"values/{self.data_name}/representations/{self.name}_{self.feature_num}.pt")
        self.feature_num += 1

    def hook_compute_stable_rank(self, module, input, ouput):
        output = self.preprocess_output(ouput).T
        print(f"output shape: {output.size()}")
        rank = stable_rank(output)
        print(rank)
        self.stable_ranks.append(rank)

    # singular values
    def hook_compute_singular_values(self, module, input, output):
        output = self.preprocess_output(output).T
        if output.size(1) == 1:
            return
        print(f"output shape: {output.size()} {output.dtype}")
        d = output.size(0)
        N = output.size(1)
        # skip too large layersk
        if d > 1605632:
            print(f"skip layer : {d}")
            return
        if d <= N:
            print(f"use full matrix : {output.size()}")
            output = output.to(self.classifier_device)
            cov_mat = torch.cov(output, correction=1)
            print(f"cov_mat shape: {cov_mat.size()}")
            singular_values = torch.linalg.svdvals(cov_mat).detach().cpu()

            del cov_mat
        else:
            print(f"use random projection method : {output.size()}")
            singular_values = random_projection_method(output, self.b, self.classifier_device)

        print(singular_values[:10])
        del input
        del output
        self.singular_values.append(singular_values)

    def hook_compute_NC1(self, module, input, output):
        # output : (feature, sample)
        output = self.preprocess_output(output).T

        if output.size(1) == 1:
            return
        print(f"output shape: {output.size()}")
        _, _, nc1 = computeNC1(self.labels, output)
        print(nc1)
        self.nc1_data.append(nc1)

    def hook_compute_NC2(self, module, input, output):
        # output = self.preprocess_output(output).T
        print(f"output shape: {output.size()} {output.dtype}")

        # if output.size(1) == 1:
        #     return
        # nc2 = self.computeNC2(self.labels, output)
        nc2 = average_pairwise_cosine_distance(output)
        print(nc2)

        self.nc2_data.append(nc2)

    def computeNC2(self, labels, embeddings):
        """
        Computes the NC2 condition for the embeddings, which includes the mean of the adjusted cosine similarity
        between the class means and their standard deviation.

        Parameters:
        labels (torch.Tensor): A vector of labels.
        embeddings (torch.Tensor): A d x num_samples matrix of d-dimensional embeddings.

        Returns:
        float: The mean of 1 / (1 - C) + the cosine similarity for all class mean pairs.
        float: The standard deviation of the cosine similarities for all class mean pairs.
        """
        # Calculate class means (mu_c)
        unique_classes = labels.unique()
        class_means = torch.stack([embeddings[:, labels == c].mean(dim=1) for c in unique_classes])

        # Center the class means
        mu_G = class_means.mean(dim=0)
        centered_class_means = class_means - mu_G

        # Normalize the centered class means to get the simplex ETF structure vectors
        simplex_vectors = centered_class_means / torch.norm(centered_class_means, dim=1, keepdim=True)

        # Compute pairwise cosine similarity for all class mean pairs
        cosine_similarities = torch.mm(simplex_vectors, simplex_vectors.t())

        # Exclude self-similarity by filling the diagonal with zeros
        cosine_similarities.fill_diagonal_(0)

        # Adjust the cosine similarities according to the NC2 condition
        C = unique_classes.size(0)
        adjusted_cosine_similarities = 1 / (C - 1) + cosine_similarities
        # adjusted_cosine_similarities = cosine_similarities

        # Compute mean and standard deviation for the upper triangle of the adjusted cosine matrix
        # to avoid counting pairs twice and including the diagonal
        upper_triangle_indices = torch.triu_indices(row=C, col=C, offset=1)
        mean_similarity = (
            adjusted_cosine_similarities[upper_triangle_indices[0], upper_triangle_indices[1]].mean().item()
        )
        # std_similarity = (
        #     adjusted_cosine_similarities[upper_triangle_indices[0], upper_triangle_indices[1]]
        #     .std()
        #     .item()
        # )
        print(mean_similarity.dtype)
        return mean_similarity

    def hook_compute_nearest_mean_prediction(self, module, input, output):
        output = self.preprocess_output(output)
        print(f"output shape: {output.size()}")
        if output.size(1) == 1:
            return

        class_means, _ = self.get_class_means_nums(output, self.labels)
        prediction = self.nearest_class_mean_classifier(output, class_means)
        del output
        self.closest_class_predictions.append(prediction)

    def get_class_means_nums(self, output, labels):
        class_means = []
        class_nums = []
        if self.data_name == "imagenet":
            class_counts = 1000
        elif self.data_name == "cifar10":
            class_counts = 10
        elif self.data_name == "places":
            class_counts = 365
        elif self.data_name == "imagenet100":
            class_counts = 100
        for i in range(class_counts):
            means = torch.mean(output[labels == i], dim=0)
            class_means.append(torch.mean(output[labels == i], dim=0))
            class_nums.append(torch.sum(labels == i))
        return class_means, class_nums

    def nearest_class_mean_classifier(self, embeddings, class_means):
        distances = euclidean_distances(embeddings, class_means)
        prediction = np.argmin(distances, axis=1)
        print(f"prediction shape: {prediction.shape}")
        return prediction

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

    def inspect_layers_dim(self, sample_size=15000, dummy_input=None, preprocess=False):
        # self.register_full_hooks()
        self.register_hooks(self.hook_full)
        if dummy_input is None:
            self.forward(self.dummy_data.to(self.main_device))
        else:
            print("use preprocess dummy input")
            self.forward(dummy_input.to(self.main_device))

        for representation in self.representations:
            dim = representation.view(representation.size(0), -1).size(1)
            # representation_bytes = representation.element_size() * dim * sample_size
            # g_bytes = representation.element_size() * dim * self.b
            # reduced_bytes = representation.element_size() * self.b * sample_size
            # total_bytes = representation_bytes + g_bytes + reduced_bytes
            # size = get_size(total_bytes)
            # print(f"layer dim: {dim}, size: {size} GB")
            print(f"layer dim: {dim}")

        self.remove_hooks()

    def init_classifers(self, resolution):
        self.classifiers = []
        self.optimizers = []
        dummy = torch.randn(1, 3, resolution, resolution)
        self.forward(dummy.to(self.main_device))
        if "cifar" in self.data_name:
            num_classes = 10
        else:
            num_classes = 1000

        for i, representation in enumerate(self.representations):
            cur_classifier = nn.Linear(representation.size(1), num_classes)

            cur_optim = optim.Adam(cur_classifier.parameters(), lr=self.lr)
            cur_classifier.to(self.classifier_device)
            self.classifiers.append(cur_classifier)
            self.optimizers.append(cur_optim)
        print(f"classifier counts: {len(self.classifiers)}")

    def check_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def set_lr(self, lr):
        self.lr = lr

    @abstractmethod
    def preprocess_output(self, output) -> torch.FloatTensor:
        pass

    def forward(self, input):
        # reset representations
        self.singular_values = []
        self.representations = []
        self.closest_class_predictions = []
        self.class_means = []
        input = input.to(self.main_device)
        output = self.model(input)
        return output

    def download_stable_rank(self, input_data, specific_data_name):
        self.model.to(self.main_device)
        self.stable_ranks = []

        self.register_hooks(self.hook_compute_stable_rank)
        base_path = f"values/{self.data_name}/stable_rank"
        singular_save_path = f"{base_path}/{specific_data_name}"

        self.check_folder(singular_save_path)
        self.forward(input_data)

        # check is the folder exist
        torch.save(
            self.stable_ranks,
            f"{singular_save_path}/{self.name}.pt",
        )
        self.remove_hooks()

    def download_singular_values(self, input_data, specific_data_name, GAP=False, pretrained=True):
        self.model.to(self.main_device)
        if not GAP:
            self.register_hooks(self.hook_compute_singular_values)
            base_path = f"values/{self.data_name}/singular_values"
        else:
            self.register_hooks(self.hook_compute_vectorized_singular_values())
            base_path = f"values/{self.data_name}/singular_values_GAP"

        singular_save_path = f"{base_path}/{specific_data_name}"

        self.check_folder(singular_save_path)
        self.forward(input_data)

        # check is the folder exist
        torch.save(
            self.singular_values,
            f"{singular_save_path}/{self.name}{'_random_init' if not pretrained else ''}.pt",
        )
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

    def add_gpus(self, main_device, classifier_device):
        self.main_device = main_device
        self.model.to(self.main_device)
        self.classifier_device = classifier_device

    def download_accuarcy(self, train_dataloader, test_dataloader, pretrained_data, resolution, GAP=True):
        # create folder
        save_path = (
            f"values/{pretrained_data}/{'acc' if not self.OOD else f'ood_acc/{self.data_name}'}/{self.name}"
        )
        self.check_folder(save_path)

        # register hooks
        if not GAP:
            print("use full hook")
            self.register_hooks(self.hook_full)

        else:
            self.register_hooks(self.hook_vectorization(resolution=resolution))

        # init classifiers
        self.init_classifers(resolution)
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

    def check_last_layer_acc(self, train_dataloader, test_dataloader, resolution):
        self.register_hooks(self.hook_vectorization(resolution=resolution), only_last_layer=True)
        self.init_classifers(resolution)
        self.train_classifers(train_dataloader, test_dataloader)
        acc = self.test_classifers(test_dataloader)[0]
        return acc

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
                # for data, target in data_loader:
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

    def test_classifers(self, test_loader):
        self.model.eval()
        for classifier in self.classifiers:
            classifier.eval()
        accuracies = [0 for _ in range(len(self.classifiers))]
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.main_device), target.to(self.classifier_device)
                self.forward(data)
                total += len(target)

                for idx, (representation, classifier) in enumerate(
                    zip(self.representations, self.classifiers)
                ):
                    representation = representation.to(self.classifier_device)
                    output = classifier(representation)
                    # loss = self.criterion(output, target)
                    correct = output.argmax(dim=1).eq(target).sum().item()
                    accuracies[idx] += correct
        accuracies = [round(100.0 * correct / total, 2) for correct in accuracies]
        return accuracies
        # plot accuracy

    def download_NC1(self, features, labels):
        self.labels = labels
        self.nc1_data = []
        base_path = f"values/{self.data_name}/NC1"
        self.check_folder(base_path)
        self.register_hooks(self.hook_compute_NC1)

        _ = self.forward(features)
        torch.save(self.nc1_data, f"{base_path}/{self.name}.pt")

    def download_NC4(self, features, labels):
        self.labels = labels
        base_path = f"values/{self.data_name}/NC4"
        self.check_folder(base_path)

        self.closest_class_predictions = []
        self.register_hooks(self.hook_compute_nearest_mean_prediction)
        output = self.forward(features)
        dnn_prediction = output.argmax(dim=1).detach().cpu().numpy()
        accuracies = []
        for prediction in self.closest_class_predictions:
            acc = np.mean(prediction == dnn_prediction)
            print(acc)
            accuracies.append(acc)
        torch.save(accuracies, f"{base_path}/{self.name}.pt")

    def download_NC2(self, features, labels):
        self.labels = labels
        base_path = f"values/{self.data_name}/NC2"
        self.check_folder(base_path)
        self.nc2_data = []
        self.register_hooks(self.hook_compute_NC2)
        self.forward(features)

        torch.save(self.nc2_data, f"{base_path}/{self.name}.pt")

    def check_simplex_etf_structure(self, class_means: np.ndarray) -> (bool, float):
        # class_means : (num_classes, num_features)
        num_classes = class_means.shape[0]
        global_mean = np.mean(class_means, axis=0)

        # Centering the class means
        centered_class_means = class_means - global_mean

        # Normalizing centered class means to unit length
        normalized_class_means = centered_class_means / np.linalg.norm(
            centered_class_means, axis=1, keepdims=True
        )

        # Dot product matrix
        dot_product_matrix = np.dot(normalized_class_means, normalized_class_means.T)

        # Extract the lower triangular part of the matrix, excluding the diagonal
        lower_triangular = dot_product_matrix[np.tril_indices(num_classes, -1)]

        # Compute the ETF metric
        etf_metric = np.mean(lower_triangular)

        result = (1 / (1 - num_classes)) - etf_metric
        return result


class VGGAnalyzer(Analyzer):
    def __init__(self, model, model_name, dummy_input):
        super().__init__(model, model_name, dummy_input)

    def get_layers(self):
        layers = []
        for layer in self.model.features:
            if isinstance(layer, nn.Conv2d):
                layers.append(layer)
        return layers

    def preprocess_output(self, output) -> torch.FloatTensor:
        output = output.view(output.size(0), -1)
        return output


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

    def get_layers(self):
        layers = []
        for name, module in self.model.named_modules():
            if "conv" in name and isinstance(module, nn.Conv2d):
                layers.append(module)
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
                if "conv" in name:
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
        print(output.size())
        output = output.view(output.size(0), -1)
        return output


class AVITAnalyzer(Analyzer):
    def __init__(self, model, model_name, data_name):
        super().__init__(model, model_name, data_name)

    def get_layers(self):
        layers = []
        for module in self.model.modules():
            if isinstance(module, Block_ACT):
                layers.append(module)
        return layers

    def preprocess_output(self, output):
        output = output[:, 1:]
        output = output.view(output.size(0), -1)
        return output


class VITAnalyzer(Analyzer):
    def __init__(self, model, model_name, data_name):
        super().__init__(model, model_name, data_name)

    def get_layers(self):
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, Block):
                layers.append(module)
        print(len(layers))
        return layers

    def preprocess_output(self, output):
        output = output[:, 1:]
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
    elif "vgg" in model_name:
        return VGGAnalyzer(model, model_name, dummy_input)

    elif "resnet" in model_name:
        return ResNetAnalyzer(model, model_name, dummy_input)
    elif "convnextv2" in model_name.lower():
        return ConvNextV2Analyzer(model, model_name, dummy_input)
    elif "avit" in model_name.lower():
        return AVITAnalyzer(model, model_name, dummy_input)

    elif "vit" in model_name.lower():
        return VITAnalyzer(model, model_name, dummy_input)
    elif "mugs" in model_name.lower():
        return VITAnalyzer(model, model_name, dummy_input)
    elif "mae" in model_name.lower():
        return VITAnalyzer(model, model_name, dummy_input)
    elif "convnext" in model_name.lower():
        return ConvNextAnalyzer(model, model_name, dummy_input)
    elif "swin" in model_name.lower():
        print(f"loading {model_name} analyzer")
        return SwinAnalyzer(model, model_name, dummy_input)
    elif "dinov2" in model_name.lower():
        print(f"loading {model_name} analyzer")
        return DINOV2Analyzer(model, model_name, dummy_input)
    elif "dino" in model_name.lower():
        return VITAnalyzer(model, model_name, dummy_input)
    else:
        raise ValueError("model name not supported")
