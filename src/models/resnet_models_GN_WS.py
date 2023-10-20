import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#           'resnet152']
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


class Conv2d(nn.Conv2d):  # For Weight Standardization
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = (
            weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        )
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.GroupNorm(32, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.GroupNorm(32, planes)
        self.downsample = downsample
        self.stride = stride

        gn_init(self.bn1)
        gn_init(self.bn2, zero_init=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(32, planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, planes)
        self.conv3 = Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(32, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        gn_init(self.bn1)
        gn_init(self.bn2)
        gn_init(self.bn3, zero_init=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv2d_init(m):
    assert isinstance(m, nn.Conv2d)
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2.0 / n))


def gn_init(m, zero_init=False):
    assert isinstance(m, nn.GroupNorm)
    m.weight.data.fill_(0.0 if zero_init else 1.0)
    m.bias.data.zero_()


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv2d_init(m)
        gn_init(self.bn1)

        # for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #    elif isinstance(m, nn.BatchNorm2d):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)

        # for m in self.modules():
        #    if isinstance(m, Bottleneck):
        #        print("INIT BN ", m)
        #        nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(32, planes * block.expansion),
            )
            m = downsample[1]
            assert isinstance(m, nn.GroupNorm)
            gn_init(m)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model


class ResNet18_StartAt_Layer4_1(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_Layer4_1, self).__init__()

        self.model = resnet18(pretrained=False)
        if num_classes is not None:
            print("Changing output layer to contain %d classes." % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1
        del self.model.layer2
        del self.model.layer3
        del self.model.layer4[0]

    def forward(self, x):
        out = self.model.layer4(x)
        out = F.avg_pool2d(out, out.size()[3])
        final_embedding = out.view(out.size(0), -1)
        out = self.model.fc(final_embedding)
        return out


class ResNet18_StartAt_Layer4_0(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_Layer4_0, self).__init__()

        self.model = resnet18(pretrained=False)
        if num_classes is not None:
            print("Changing output layer to contain %d classes." % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1
        del self.model.layer2
        del self.model.layer3

    def forward(self, x):
        out = self.model.layer4(x)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.model.fc(out)
        return out


## Block-4
class BaseResNet18ClassifyAfterLayer4(nn.Module):
    def __init__(self, num_del=0, num_classes=None):
        super(BaseResNet18ClassifyAfterLayer4, self).__init__()
        self.model = resnet18(pretrained=False)
        for _ in range(0, num_del):
            del self.model.layer4[-1]
        if num_classes is not None:
            print("Changing num_classes to {}".format(num_classes))
            self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out


class ResNet18ClassifyAfterLayer4_0(BaseResNet18ClassifyAfterLayer4):
    def __init__(self, num_classes=None):
        super(ResNet18ClassifyAfterLayer4_0, self).__init__(num_del=1, num_classes=num_classes)


class ResNet18ClassifyAfterLayer4_1(BaseResNet18ClassifyAfterLayer4):
    def __init__(self, num_classes=None):
        super(ResNet18ClassifyAfterLayer4_1, self).__init__(num_del=0, num_classes=num_classes)


# net = eval('ResNet18ClassifyAfterLayer4_1')()
# x = torch.randn((10, 3, 224, 224), dtype=torch.float32)
# y = net(x)
# print('shape of y:', y.shape)


class ResNet18_StartAt_Layer3_1(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_Layer3_1, self).__init__()

        self.model = resnet18(pretrained=False)
        if num_classes is not None:
            print("Changing output layer to contain %d classes." % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1
        del self.model.layer2
        del self.model.layer3[0]

    def forward(self, x):
        out = self.model.layer3(x)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.model.fc(out)
        return out


class ResNet18_StartAt_Layer3_0(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_Layer3_0, self).__init__()

        self.model = resnet18(pretrained=False)
        if num_classes is not None:
            print("Changing output layer to contain %d classes." % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1
        del self.model.layer2

    def forward(self, x):
        out = self.model.layer3(x)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.model.fc(out)
        return out


# Block-3
class BaseResNet18ClassifyAfterLayer3(nn.Module):
    def __init__(self, num_del=0, num_classes=None):
        super(BaseResNet18ClassifyAfterLayer3, self).__init__()
        self.model = resnet18(pretrained=False)

        for _ in range(0, num_del):
            del self.model.layer3[-1]
        if num_classes is not None:
            print("Changing num_classes to {}".format(num_classes))
            self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out


class ResNet18ClassifyAfterLayer3_0(BaseResNet18ClassifyAfterLayer3):
    def __init__(self, num_classes=None):
        super(ResNet18ClassifyAfterLayer3_0, self).__init__(num_del=1, num_classes=num_classes)


class ResNet18ClassifyAfterLayer3_1(BaseResNet18ClassifyAfterLayer3):
    def __init__(self, num_classes=None):
        super(ResNet18ClassifyAfterLayer3_1, self).__init__(num_del=0, num_classes=num_classes)


###


class ResNet18_StartAt_Layer2_1(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_Layer2_1, self).__init__()

        self.model = resnet18(pretrained=False)
        if num_classes is not None:
            print("Changing output layer to contain %d classes." % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1
        del self.model.layer2[0]

    def forward(self, x):
        out = self.model.layer2(x)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.model.fc(out)
        return out


class ResNet18_StartAt_Layer2_0(nn.Module):
    def __init__(self, num_classes=None):
        super(ResNet18_StartAt_Layer2_0, self).__init__()

        self.model = resnet18(pretrained=False)
        if num_classes is not None:
            print("Changing output layer to contain %d classes." % num_classes)
            self.model.fc = nn.Linear(512, num_classes)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1

    def forward(self, x):
        out = self.model.layer2(x)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.model.fc(out)
        return out


# Block-2
class BaseResNet18ClassifyAfterLayer2(nn.Module):
    def __init__(self, num_del=0, num_classes=None):
        super(BaseResNet18ClassifyAfterLayer2, self).__init__()
        self.model = resnet18(pretrained=False)

        for _ in range(0, num_del):
            del self.model.layer2[-1]
        if num_classes is not None:
            print("Changing num_classes to {}".format(num_classes))
            self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out


class ResNet18ClassifyAfterLayer2_0(BaseResNet18ClassifyAfterLayer2):
    def __init__(self, num_classes=None):
        super(ResNet18ClassifyAfterLayer2_0, self).__init__(num_del=1, num_classes=num_classes)


class ResNet18ClassifyAfterLayer2_1(BaseResNet18ClassifyAfterLayer2):
    def __init__(self, num_classes=None):
        super(ResNet18ClassifyAfterLayer2_1, self).__init__(num_del=0, num_classes=num_classes)


###
