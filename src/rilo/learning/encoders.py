"""Compact shared RGB encoder with spatial keypoints and measured proprioception.

ResNet-18 topology follows the original residual architecture. GroupNorm avoids
batch-size-dependent statistics. ImageNet initialization is an explicit training
option; checkpoint loading and inference never download weights.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from rilo.learning.data import ObservationSpec

IMAGENET_RESNET18_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def group_norm(channels):
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualImageBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False)
        self.bn1 = group_norm(outgoing)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.bn2 = group_norm(outgoing)
        self.downsample = (nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False),
                                        group_norm(outgoing)) if incoming != outgoing or stride != 1 else None)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + residual)


class ResNet18GN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = group_norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._layer(64, 64, 1)
        self.layer2 = self._layer(64, 128, 2)
        self.layer3 = self._layer(128, 256, 2)
        self.layer4 = self._layer(256, 512, 2)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    @staticmethod
    def _layer(incoming, outgoing, stride):
        return nn.Sequential(ResidualImageBlock(incoming, outgoing, stride), ResidualImageBlock(outgoing, outgoing))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

    def initialize_imagenet(self):
        weights = torch.hub.load_state_dict_from_url(IMAGENET_RESNET18_URL, map_location="cpu", check_hash=True)
        usable = {key: value for key, value in weights.items()
                  if key in self.state_dict() and self.state_dict()[key].shape == value.shape}
        self.load_state_dict(usable, strict=True)


class SpatialKeypoints(nn.Module):
    def __init__(self, incoming, keypoints=32):
        super().__init__()
        self.projection = nn.Conv2d(incoming, keypoints, 1)

    def forward(self, x):
        logits = self.projection(x)
        batch, channels, height, width = logits.shape
        probabilities = logits.flatten(2).softmax(-1)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype),
                                torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype), indexing="ij")
        coordinates = torch.stack((xx.flatten(), yy.flatten()), -1)
        return (probabilities @ coordinates).reshape(batch, channels*2)


class ObservationEncoder(nn.Module):
    def __init__(self, spec: ObservationSpec, feature_dim=128, keypoints=32, backbone="resnet18"):
        super().__init__()
        self.spec, self.feature_dim, self.backbone_name = spec, feature_dim, backbone
        if backbone == "resnet18":
            self.backbone = ResNet18GN()
            channels = 512
        elif backbone == "tiny":
            self.backbone = nn.Sequential(nn.Conv2d(3, 16, 5, 2, 2), group_norm(16), nn.SiLU(),
                nn.Conv2d(16, 32, 3, 2, 1), group_norm(32), nn.SiLU(),
                nn.Conv2d(32, 64, 3, 2, 1), group_norm(64), nn.SiLU())
            channels = 64
        else:
            raise ValueError("Encoder backbone must be resnet18 or tiny (smoke tests)")
        self.keypoints = SpatialKeypoints(channels, keypoints)
        self.fusion = nn.Sequential(nn.Linear(len(spec.cameras)*keypoints*2 + spec.state_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.SiLU())
        self.register_buffer("state_mean", torch.zeros(spec.state_dim))
        self.register_buffer("state_std", torch.ones(spec.state_dim))
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def set_state_statistics(self, mean, std):
        self.state_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.state_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-3))

    def initialize_imagenet(self):
        if self.backbone_name != "resnet18":
            raise ValueError("ImageNet initialization requires the ResNet-18 backbone")
        self.backbone.initialize_imagenet()

    def forward(self, images, states):
        # Leading dimensions may be [B] or [B, observation_history].
        leading = images.shape[:-4]
        cameras, channels, height, width = images.shape[-4:]
        if cameras != len(self.spec.cameras) or channels != 3:
            raise ValueError("Camera tensor does not match the checkpoint representation")
        pixels = images.reshape(-1, 3, height, width)
        pixels = (pixels-self.rgb_mean)/self.rgb_std
        features = self.keypoints(self.backbone(pixels)).reshape(*leading, -1)
        state = ((states-self.state_mean)/self.state_std).clamp(-10, 10)
        return self.fusion(torch.cat((features, state), dim=-1))
