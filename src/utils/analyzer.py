from abc import ABCMeta, abstractmethod

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
from torch import nn, optim
from tqdm import tqdm
from welford import Welford


class Analyzer(metaclass=ABCMeta):
    def __init__(self, model, name, dummy_input):
        self.dummy_data = dummy_input
        self.skip_layers = False
        self.skip_numbers = 2
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.name = name
        self.total_samples = 0
        # self.device = "cpu"
        # self.device = torch.device("mps")
        for param in model.parameters():
            param.requires_grad = False

        self.model = model

        self.criterion = nn.CrossEntropyLoss()
        self.representations = []

        self.init_setup()

    def init_setup(self):
        self.layers = self.get_layers()
        self.register_hooks()
        self.init_classifers()
        self.init_storage()

    @abstractmethod
    def get_layers(self) -> list[nn.Module]:
        pass

    def register_hooks(self):
        for layer in self.layers:
            layer.register_forward_hook(self.hook_fn)

    def init_storage(self):
        self.vars = []
        self.means = []
        for representation in self.representations:
            cur_vars = Welford()
            self.vars.append(cur_vars)
            self.means.append(torch.zeros(representation.size(1)))

    def init_classifers(self):
        self.classifiers = []
        self.optimizers = []
        dummy_data = torch.zeros((1, 3, 32, 32)).to(self.device)
        self.forward(dummy_data)

        for representation in self.representations:
            cur_classifier = nn.Linear(representation.size(1), 100)

            cur_optim = optim.Adam(cur_classifier.parameters(), lr=0.001)
            cur_classifier.to(self.device)
            self.classifiers.append(cur_classifier)
            self.optimizers.append(cur_optim)

    def hook_fn(self, module, input, output):
        output = self.preprocess_output(output).detach().cpu()
        # output = self.preprocess_output(output)
        self.representations.append(output)

    @abstractmethod
    def preprocess_output(self, output) -> torch.FloatTensor:
        pass

    def forward(self, input):
        # reset representations
        self.representations = []
        _ = self.model(input)

    def download_values(self, test_dataloader, input_data, input_size):
        self.download_singular_values(input_data)
        self.download_sig_count(test_dataloader, input_data, input_size)

    def download_accuarcy(self, train_data_loader, test_dataloader):
        self.train_classifers(train_data_loader)
        self.test_classifers(test_dataloader)

    # compute online variances
    def download_online_variances(self, test_dataloader, sample_size):
        self.calculate_online_variance(test_dataloader, sample_size)
        variances = self.get_online_variance()
        torch.save(variances, f"values/vars/{self.name}.pt")

    def download_means(self, input_data):
        self.forward(input_data)
        means = []
        for representation in self.representations:
            mean = torch.mean(representation, dim=0)
            means.append(mean)
        torch.save(means, f"values/means/{self.name}_t.pt")

    def download_online_means(self, test_dataloader, sample_size):
        self.calculate_online_mean(test_dataloader, sample_size)
        torch.save(self.means, f"values/means/{self.name}.pt")

    def get_online_variance(self):
        variances = []
        for cur_vars in self.vars:
            variances.append(torch.tensor(cur_vars.var_s))
        return variances

    def calculate_online_variance(self, data_loader, sample_size):
        count = 0

        for idx, (data, target) in enumerate(data_loader):
            count += data.size(0)
            print(f"processing batch {idx} : {count} / {sample_size}")
            data, target = data.to(self.device), target.to(self.device)
            self.online_variance_update(data)
            if count >= sample_size:
                break

    def calculate_online_mean(self, data_loader, sample_size):
        count = 0

        for idx, (data, target) in enumerate(data_loader):
            count += data.size(0)
            print(f"processing batch {idx} : {count} / {sample_size}")
            data, target = data.to(self.device), target.to(self.device)
            self.online_mean_update(data)
            if count >= sample_size:
                break

    def online_variance_update(self, input):
        self.forward(input)
        for representation, cur_vars in zip(self.representations, self.vars):
            cur_vars.add_all(np.array(representation.detach().cpu()))

    def online_mean_update(self, input):
        self.forward(input)
        for idx, representation in enumerate(self.representations):
            batch_size = representation.size(0)
            batch_mean = torch.sum(representation, dim=0) / batch_size
            self.means[idx] = (self.total_samples * self.means[idx] + batch_size * batch_mean) / (
                self.total_samples + batch_size
            )

        self.total_samples += representation.size(0)

    # compute download sigular values
    def download_singular_values(self, input_data):
        print("start download singular values")
        sigs = []
        self.forward(input_data)
        for representation in self.representations:
            cur_representation = representation.T
            if cur_representation.size(0) > 8000:
                cur_representation = cur_representation[:8000]
            print(cur_representation.size())
            cov_mat = torch.cov(cur_representation, correction=1)
            print(cov_mat.size())
            sig = torch.svd(cov_mat).S
            sigs.append(sig)
        torch.save(sigs, f"values/sigs/{self.name}.pt")

    def download_representations(self, input_data):
        self.forward(input_data)
        torch.save(self.representations, f"values/representations/{self.name}.pt")

    def download_qr_sigs_values(self, input_data):
        print("start download qr_sig values")
        sigs = []
        self.forward(input_data)
        for representation in self.representations:
            cur_representation = representation.T
            if cur_representation.size(0) > 8000:
                cur_representation = cur_representation[:8000]
            print(f"cur representation size: {cur_representation.size()}")
            sig = self.qr_svd_method(cur_representation)
            print(f"sig size: {sig.size()}")

            sigs.append(sig)
        torch.save(sigs, f"values/qr_sigs/{self.name}.pt")

    def qr_svd_method(self, X_prime):
        Q, R = torch.linalg.qr(X_prime.T)
        U, S, V = torch.svd(R)
        S = torch.sort(S, descending=True).values
        S_squared = S**2
        S_normalized = S_squared / torch.sum(S_squared)
        return S_normalized.detach()

    def download_sig_count(self, data_loader, input_data, sample_size):
        self.download_online_variances(data_loader, sample_size)
        sigs_counts = []
        self.forward(input_data)
        for representation, cur_vars in zip(self.representations, self.vars):
            cur_representation = representation.T
            if cur_representation.size(0) > 8000:
                cur_representation = cur_representation[::2]
            print(cur_representation.size())
            cov_mat = torch.cov(cur_representation, correction=1)
            print(cov_mat.size())
            vars = torch.tensor(cur_vars.var_s)
            total_vars = torch.sum(vars)
            count = self.get_count_90svd(cov_mat, total_vars)
            sigs_counts.append(count)
        torch.save(sigs_counts, f"values/sigs_count/{self.name}.pt")

    def get_count_90svd(self, X, total_vars):
        sigs = self.get_randomized_svd(X)
        total_sum = torch.sum(sigs)
        if total_sum > total_vars:
            print("something wrong")
        else:
            print("pass")
        cum_sum = torch.cumsum(sigs, dim=0)
        count = (cum_sum / total_sum <= 0.9).sum().item()
        return count

    def get_randomized_svd(self, X: torch.tensor):
        X_np = X.numpy()
        # dask_array = da.from_array(X_np, chunks=(chunk_size, chunk_size))
        n_components = min(X.size(0) - 1, 1000) - 1
        U, s, VT = randomized_svd(X_np, n_components=n_components)
        # u, s, v = da.linalg.svd_compressed(dask_array, k=n_components)
        return torch.tensor(s)

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

    def test_classifers(self, test_loader):
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
                    loss = self.criterion(output, target)
                    correct = output.argmax(dim=1).eq(target).sum().item()
                    accuracies[idx] += correct
        accuracies = [
            round(100.0 * correct / len(test_loader.dataset), 2) for correct in accuracies
        ]
        print(f"accuracy: {accuracies}")
        torch.save(accuracies, f"values/acc/{self.name}.pt")
        # plot accuracy

    def download_variances(self):
        variances = []
        for idx, representation in enumerate(self.representations):
            print(f"processing layer {idx}")
            var = torch.var(representation, dim=0)
            variances.append(var)
        torch.save(variances, f"vars/{self.name}.pt")

    def analyze_rank(self):
        sigs = self.get_sigular_values()
        thresholds = [1, 10, 20, 30, 40, 50, 100, 300, 500, 1000, 1500, 2000, 3000, 5000, 10000]
        for threshold in thresholds:
            ranks = []
            for sig in sigs:
                rank = (sig > threshold).sum().item()
                ranks.append(rank)
            self.plot_ranks(threshold, ranks)
            print(f"pretrained_ranks{threshold} = {ranks}")

    def get_sigular_values(self):
        singular_values = []
        for representation in self.representations:
            cur_representation = representation.T
            if cur_representation.size(0) > 8000:
                cur_representation = cur_representation[:8000]
            print(cur_representation.size())
            cov_mat = torch.cov(cur_representation, correction=1)
            # variances = torch.var(representation, dim=1)

            # Create a diagonal matrix using the variances

            # cov_mat = torch.diag(variances)
            _, singular_value, _ = torch.svd(cov_mat)
            # singular_value = torch.linalg.svdvals(cov_mat)
            singular_values.append(singular_value)
        return singular_values

    def plot_ranks(self, threshold, ranks):
        plt.figure()
        plt.plot(ranks)
        plt.xlabel("layers")
        # set limit for x axis 0
        plt.ylim(bottom=0)
        plt.ylabel("rank")
        plt.title(f"Resnet34 reg rank - {threshold} threshold")
        plt.savefig(f"ranks_{threshold}.png")


class ConvAnalyzer(Analyzer):
    def __init__(self, model):
        super().__init__(model)

    def get_layers(self) -> list[nn.Module]:
        layers = []
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                layers.append(layer)
        return layers

    def preprocess_output(self, output) -> torch.FloatTensor:
        # current input shape is (batch_size, channels, height, width) I want (channels * height * width, batch_size)
        output = output.view(output.size(0), -1)
        return output


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
