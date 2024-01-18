import math
import sys
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial

sys.path.append("A-ViT")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize
from timm import create_model
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Block, VisionTransformer
from torchvision import models
from torchvision.models import resnet34

import models.vit_models
from models.resnet18 import get_resnet18
from models.vgg import VGG

# from models.resnet_models_GN_WS import resnet34


class ResNet(nn.Module, metaclass=ABCMeta):
    def __init__(self, device="cpu", weights_path=None):
        """ResNet34 model for CIFAR-10

        Args:
            weights_path (str): Path to pretrained weights
            device (str): Device to load weights onto
        """
        super().__init__()
        self.init_model()
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.resnet.maxpool = nn.Identity()
        # self.resnet.bn1 = nn.BatchNorm2d(64)
        # self.resnet.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.resnet.fc = nn.Linear(512, 10)
        if weights_path:
            if "swav" in weights_path:
                print("loading swav model")
                state = torch.load(weights_path)["state_dict"]
                for k in list(state.keys()):
                    if "backbone" in k:
                        state[k.replace("backbone.", "")] = state[k]
                    del state[k]
                self.resnet.load_state_dict(state, strict=False)
            else:
                print("loading pretrained model")
                self.resnet.load_state_dict(torch.load(weights_path, map_location=device))
        self.resnet.to(device)
        self.resnet.eval()

    @abstractmethod
    def init_model(self):
        # self.resnet = models.resnet34(weights=None)
        pass

    def forward(self, x):
        return self.resnet(x)


class ResNet18(ResNet):
    def init_model(self):
        self.resnet = models.resnet18(weights=None)


class ResNet34(ResNet):
    def init_model(self):
        self.resnet = models.resnet34(weights=None)


class ResNet34_GN(ResNet):
    def init_model(self):
        self.resnet = resnet34()


class MLP(nn.Module):
    def __init__(self, device="cuda", weights_path=None):
        super(MLP, self).__init__()
        layers = []
        input_size = 32 * 32 * 3  # CIFAR-10 images are 32x32 pixels with 3 color channels

        layers = []
        layers.append(nn.Linear(input_size, 1028))
        layers.append(nn.ReLU())

        for _ in range(11):  # 11 layers in total (1 to 12)
            layers.append(nn.Linear(1028, 1028))
            layers.append(nn.ReLU())

        # 13
        layers.append(nn.Linear(1028, 10))  # 10 output classes for CIFAR-10

        # layer init
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        self.layers = nn.Sequential(*layers)

        if weights_path:
            self.load_state_dict(torch.load(weights_path, map_location=device))

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.layers(x)
        return x


class IterativeKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.nn_model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.data = None
        self.labels = None
        self.train_data = None
        self.predicted_labels = None
        self.target_labels = None

    def update(self, new_data, new_labels):
        new_data = np.array(new_data)
        new_labels = np.array(new_labels)
        if self.data is None:
            self.data = new_data
            self.labels = new_labels
        else:
            self.data = np.concatenate((self.data, new_data))
            self.labels = np.concatenate((self.labels, new_labels))

    def train(self):
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        X_train = (self.data - self.mean) / self.std
        X_train = normalize(X_train, axis=1)

        # Normalization to unit length
        self.nn_model.fit(X_train, self.labels)

    def predict(self, X, target_labels):
        X = (X - self.mean) / self.std
        X = normalize(X, axis=1)

        labels = self.nn_model.predict(X)
        if self.target_labels is None:
            self.target_labels = target_labels
        else:
            self.target_labels = np.concatenate((self.target_labels, target_labels))
        if self.predicted_labels is None:
            self.predicted_labels = labels
        else:
            self.predicted_labels = np.concatenate((self.predicted_labels, labels))

    def get_accuracy(self):
        return np.sum(self.target_labels == self.predicted_labels) / len(self.target_labels)


def convnextv2_fcmae(pretrained=True):
    model = convnextv2_base()

    if pretrained:
        finetune = "weights/convnextv2_base_1k_224_fcmae.pt"
        checkpoint = torch.load(finetune, map_location="cpu")

        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # remove decoder weights
        checkpoint_model_keys = list(checkpoint_model.keys())
        for k in checkpoint_model_keys:
            if "decoder" in k or "mask_token" in k or "proj" in k or "pred" in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        checkpoint_model = remap_checkpoint_keys(checkpoint_model)
        load_state_dict(model, checkpoint_model)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
        torch.nn.init.constant_(model.head.bias, 0.0)
    return model


def remap_checkpoint_keys(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith("encoder"):
            k = ".".join(k.split(".")[1:])  # remove encoder in the name
        if k.endswith("kernel"):
            k = ".".join(k.split(".")[:-1])  # remove kernel in the name
            new_k = k + ".weight"
            if len(v.shape) == 3:  # resahpe standard convolution
                kv, in_dim, out_dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(2, 1, 0).reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
            elif len(v.shape) == 2:  # reshape depthwise convolution
                kv, dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(1, 0).reshape(dim, 1, ks, ks).transpose(3, 2)
            continue
        elif "ln" in k or "linear" in k:
            k = k.split(".")
            k.pop(-2)  # remove ln and linear in the name
            new_k = ".".join(k)
        else:
            new_k = k
        new_ckpt[new_k] = v

    # reshape grn affine parameters and biases
    for k, v in new_ckpt.items():
        if k.endswith("bias") and len(v.shape) != 1:
            new_ckpt[k] = v.reshape(-1)
        elif "grn" in k:
            new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
    return new_ckpt


def load_state_dict(model, state_dict, prefix="", ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )
    if len(error_msgs) > 0:
        print("\n".join(error_msgs))


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        head_init_scale=1.0,
    ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


image100_pretrained_weights = {
    32: "0.01_0.0/32_20_41.74.pth",
    # 32: "0.01_0.0/32_31_45.98.pth",
    64: "0.01_0.005/64_10_49.84.pth",
    # 64: "0.01_0.005/64_20_58.28.pth",
    128: "0.01_0.001/128_10_57.24.pth",
    224: "0.01_0.001/224_10_64.24.pth",
    # "down_up": "0.01_0.001/224_15_53.48.pth",
    # "down_up" : "0.005_0.001/224_31_58.7.pth"
    "down_up": "0.005_0.01/224_20_55.6.pth",
}


def get_vgg13_imagenet100(model_name, pretrained=True):
    if "32" in model_name:
        resolution_size = 32
    elif "64" in model_name:
        resolution_size = 64
    elif "128" in model_name:
        resolution_size = 128
    elif "224" in model_name:
        resolution_size = 224
    elif "down_up" in model_name:
        resolution_size = "down_up"
    else:
        raise ValueError("resolution not found")
    model = VGG("VGG13")
    if pretrained:
        print(f"loading vgg13 imagenet100 {resolution_size} pretrained")
        file_name = image100_pretrained_weights[resolution_size]
        print(f"loading {file_name}")
        state_dict = torch.load(f"weights/vgg13/{resolution_size}/{file_name}")

        # remove module. prefix
        if "down_up" in model_name:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.` prefix
                new_state_dict[name] = v

        model.load_state_dict(state_dict)

    return model


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


# VGG11_pretrained_weights_by_sample_size = {
#     200: "0.001/0.0001/200.pth",  # 7
#     500: "0.001/0.0/500.pth",  # 7
#     1000: "0.001/0.0/1000.pth",  # 6
# }


Resnet34_resolution_weights = {
    32: "0.001/0.0001/32_25_52.04.pth",
    64: "0.001/0.0001/64_25_57.74.pth",
    128: "0.001/0.0001/128_20_57.7.pth",
    224: "0.001/0.0001/224_20_66.46.pth",
}

Resnet34_wide_resolution_weights = {224: "0.001/0.0001/224_30_44.78.pth"}


def get_resnet34_by_resolution(image_resolution, class_num=100, pretrained=True):
    if image_resolution not in [32, 64, 128, 224]:
        raise ValueError(f" {image_resolution} image resolution not found")

    print(f"loading resnet34 {image_resolution}")
    model = resnet34(weights=None)

    # modify layers to fit image resolution
    if image_resolution in [32, 64]:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        if image_resolution == 32:
            model.maxpool = nn.Identity()

    # change last layer
    model.fc = nn.Linear(512, class_num)

    # load pretrained weights
    if pretrained:
        print("loading wide resnet34")
        file_name = Resnet34_wide_resolution_weights[image_resolution]
        state_dict = torch.load(f"weights/resnet34_resolutions/{image_resolution}/{file_name}")
        print(f"loading {image_resolution}-{file_name}")
        model.load_state_dict(state_dict)

    return model


ResNet18_pretrained_weights_by_resolution = {
    32: "0.001/0.01/32_30_51.0.pth",
    64: "0.001/0.01/64_30_64.4.pth",
    128: "0.001/0.0/128_30_67.5.pth",
    224: "0.001/0.0/224_35_72.52.pth",
}

ResNet18_original_pretrained_weights_by_resolution = {32: "0.1/0.0001/32_final.pth"}

ResNet18_no_residual_pretrained_weights_by_resolution = {
    32: "0.001/0.01/32_30_43.84.pth",
    64: "0.001/0.001/64_40_58.46.pth",
    128: "0.001/0.001/128_40_59.68.pth",
    224: "0.001/0.001/224_40_62.92.pth",
}

ResNet18_augmentation_pretrained_weights_by_resolution = {
    32: "0.001/0.0001/32_60_61.26.pth",
    64: "0.001/0.0001/64_60_73.0.pth",
    128: "0.001/0.0001/128_70_75.68.pth",
    224: "0.001/0.0001/224_70_79.92.pth",
}


def get_resnet18_by_resolution(
    image_resolution, class_num=100, residual_connection=True, pretrained=False, augmentation=False
):
    model = get_resnet18(nclasses=class_num, residual_connection=residual_connection)
    if image_resolution in [128, 224]:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
    if pretrained:
        # file_name = ResNet18_pretrained_weights_by_resolution[image_resolution]
        if augmentation:
            print("loading augmentation model")
            file_name = ResNet18_augmentation_pretrained_weights_by_resolution[image_resolution]
            state_dict = torch.load(f"weights/resnet18_resolution_aug/{image_resolution}/{file_name}")

        elif residual_connection:
            file_name = ResNet18_original_pretrained_weights_by_resolution[image_resolution]
            state_dict = torch.load(f"weights/resnet18_resolutions_original/{image_resolution}/{file_name}")

        else:
            file_name = ResNet18_no_residual_pretrained_weights_by_resolution[image_resolution]
            state_dict = torch.load(
                f"weights/resnet18_resolutions_no_residual/{image_resolution}/{file_name}"
            )
        print(f"loading {image_resolution}-{file_name}")
        model.load_state_dict(state_dict)

    return model


VGG11_pretrained_weights_by_sample_size = {
    100: "0.001/0.0001/100_20_18.44.pth",
    200: "0.001/0.0001/200_15_25.9.pth",  # 7
    500: "0.001/0.0001/500_15_34.64.pth",
    1000: "0.001/0.0001/1000_15_40.26.pth",
}


def get_vgg11_by_sample_num(sample_per_class, pretrained=True):
    print(f"loading vgg11 sample {sample_per_class}")
    if sample_per_class not in [100, 200, 500, 1000]:
        raise ValueError("sample_per_class not found")
    model = VGG("VGG13")

    if pretrained:
        file_name = VGG11_pretrained_weights_by_sample_size[sample_per_class]
        print(f"loading {file_name}")
        state_dict = torch.load(f"weights/vgg11_100/{sample_per_class}/{file_name}")
        # state_dict = remove_module_prefix(state_dict)

        model.load_state_dict(state_dict)
    else:
        model.apply(initialize_weights)
    return model


VGG11_pretrained_weights_by_class_num = {
    10: "0.001/0.0001/10_20_62.2.pth",
    # 50: "0.001/0.0001/50_15_30.72.pth",
    50: "0.001/0.0001/50_20_31.48.pth"
    # 100: "0.001_0.01/100.pth",
}


def get_vgg11_by_class_num(class_num, pretrained=True):
    print(f"loading vgg11 class {class_num}")
    if class_num not in [10, 50, 100, 1000]:
        raise ValueError("class num not found")
    # model = models.vgg13_bn(weights=None)

    # # change last layer
    # model.classifier[6] = nn.Linear(4096, class_num)
    model = VGG("VGG13", class_num=class_num)

    # weights init
    if pretrained:
        file_name = VGG11_pretrained_weights_by_class_num[class_num]
        print(f"loading {file_name}")
        state_dict = torch.load(f"weights/vgg11_class/{class_num}/{file_name}")
        # state_dict = remove_module_prefix(state_dict)

        model.load_state_dict(state_dict)
    else:
        model.apply(initialize_weights)
    return model


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.` prefix
        new_state_dict[name] = v
    return new_state_dict


def get_vit_tiny_patch8(image_resolution):
    weights_path_dict = {
        64: "weights/vit/64.pth",
    }

    checkpoint = torch.load(weights_path_dict[image_resolution])
    args = checkpoint["args"]
    model = create_model(
        args.model,
        pretrained=False,  # args.pretrained,
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        args=args,
    )
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches**0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model["pos_embed"] = new_pos_embed
    model.load_state_dict(checkpoint_model, strict=False)
    return model
