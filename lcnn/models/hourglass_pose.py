"""
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) Yichao Zhou (LCNN)
(c) YANG, Wei
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck
import torchvision.models as models

pretrained_resnet = models.resnet50(pretrained=True)

# Adjust the first convolutional layer
# Original first conv layer: in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
# pretrained_resnet.conv1 = nn.Conv2d(in_channels=1,
#                                     out_channels=pretrained_resnet.conv1.out_channels,
#                                     kernel_size=pretrained_resnet.conv1.kernel_size,
#                                     stride=pretrained_resnet.conv1.stride,
#                                     padding=pretrained_resnet.conv1.padding,
#                                     bias=pretrained_resnet.conv1.bias)
#
# # Now the first layer will accept 1-channel inputs
# # Remember to remove the fully connected and pooling layers if they are not needed
# pretrained_resnet = nn.Sequential(*list(pretrained_resnet.children())[:-2])

# Remove the average pooling and fully connected layer
modules = list(pretrained_resnet.children())[:-2]
resnet_backbone = nn.Sequential(*modules)

__all__ = ["HourglassNet", "hg"]


class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self, head, depth, num_stacks, num_blocks, num_classes):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        block = Bottleneck
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.resnet_backbone = models.resnet50(pretrained=True)

        # Adjust the first convolutional layer to accept 1-channel input if necessary
        self.resnet_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the average pooling and fully connected layer from the ResNet
        self.resnet_backbone = nn.Sequential(*list(self.resnet_backbone.children())[:-2])

        # Add an adaptor to match the channel dimensions if necessary
        self.channel_adaptor = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # build hourglass modules
        ch = self.num_feats *  Bottleneck.expansion
        # vpts = []
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, depth))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(head(ch, num_classes))

            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1))

        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        out = []
        # out_vps = []
        x = x.repeat(1, 3, 1, 1)
        x = self.resnet_backbone(x)
        x = self.channel_adaptor(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)

            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        #
        # x = self.layer1(x)
        # x = self.maxpool(x)
        # x = self.layer2(x)
        # x = self.layer3(x)

        # for i in range(self.num_stacks):
        #     y = self.hg[i](x)
        #     y = self.res[i](y)
        #     y = self.fc[i](y)
        #     score = self.score[i](y)
        #     # pre_vpts = F.adaptive_avg_pool2d(x, (1, 1))
        #     # pre_vpts = pre_vpts.reshape(-1, 256)
        #     # vpts = self.vpts[i](x)
        #     out.append(score)
        #     # out_vps.append(vpts)
        #     if i < self.num_stacks - 1:
        #         fc_ = self.fc_[i](y)
        #         score_ = self.score_[i](score)
        #         x = x + fc_ + score_

        return out[::-1], y  # , out_vps[::-1]


def hg(**kwargs):
    resnet_layers = [3, 4, 6, 3]
    model = HourglassNet(
        head=kwargs.get("head", lambda c_in, c_out: nn.Conv2d(c_in, c_out, 1)),
        depth=kwargs["depth"],
        num_stacks=kwargs["num_stacks"],
        num_blocks=kwargs["num_blocks"],
        num_classes=kwargs["num_classes"],
    )
    return model
