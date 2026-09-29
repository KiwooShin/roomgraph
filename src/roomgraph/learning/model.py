"""Small ResNet-18 encoder/decoder with visible and typed amodal edge heads."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class Block(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing),
            nn.SiLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing),
            nn.SiLU(inplace=True),
        )


class EdgeNet(nn.Module):
    def __init__(self, pretrained=True, channels=6):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder.fc = nn.Identity()
        self.decoder = nn.ModuleList(
            [Block(512 + 256, 128), Block(128 + 128, 64), Block(64 + 64, 48), Block(48 + 64, 32)]
        )
        self.refine = Block(32 + 3, 24)
        self.head = nn.Conv2d(24, channels, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        # Preserve pretrained BN statistics on this small synthetic dataset.
        for layer in self.encoder.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, image):
        x = (image - self.mean) / self.std
        e = self.encoder
        x0 = e.relu(e.bn1(e.conv1(x)))
        x1 = e.layer1(e.maxpool(x0))
        x2 = e.layer2(x1)
        x3 = e.layer3(x2)
        x4 = e.layer4(x3)
        x = x4
        for block, skip in zip(self.decoder, [x3, x2, x1, x0], strict=True):
            x = block(
                torch.cat(
                    [
                        F.interpolate(
                            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                        ),
                        skip,
                    ],
                    dim=1,
                )
            )
        x = F.interpolate(x, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.refine(torch.cat([x, image], dim=1)))


def edge_loss(logits, target):
    target = F.max_pool2d(target.float(), 3, stride=1, padding=1)
    weighted = F.binary_cross_entropy_with_logits(
        logits.float(), target, pos_weight=torch.tensor(12.0, device=logits.device)
    )
    probability = logits.float().sigmoid()
    intersection = (probability * target).sum((0, 2, 3))
    dice = (2 * intersection + 1) / (probability.sum((0, 2, 3)) + target.sum((0, 2, 3)) + 1)
    return weighted + (1 - dice).mean()
