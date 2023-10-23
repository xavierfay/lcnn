from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lcnn.config import M


class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        for output_channels in sum(M.head_size, []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(M.head_size, []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class MultitaskLearner(nn.Module):
    def __init__(self, backbone):
        super(MultitaskLearner, self).__init__()
        self.backbone = backbone
        head_size = M.head_size
        self.num_class = sum(sum(head_size, []))
        self.head_off = np.cumsum([sum(h) for h in head_size])

    def forward(self, input_dict):
        image = input_dict["image"]
        outputs, feature = self.backbone(image)

        result = {"feature": feature}
        batch, channel, row, col = outputs[0].shape

        T = input_dict["target"].copy()
        n_jtyp = T["jmap"].shape[1]
        n_ltyp = T["lmap"].shape[1]
        print(T["jmap"].shape)


        # switch to CNHW
        for task in ["jmap", "lmap"]:
            T[task] = T[task].permute(1, 0, 2, 3)
        for task in ["joff"]:
            T[task] = T[task].permute(1, 2, 0, 3, 4)


        offset = self.head_off
        loss_weight = M.loss_weight
        losses = []
        for stack, output in enumerate(outputs):
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()
            jmap = output[0: offset[0]].reshape(n_jtyp, 2, batch, row, col)

            lmap = output[offset[0]: offset[1]].reshape(n_ltyp, 2, batch, row, col)
            joff = output[offset[1]: offset[2]].reshape(n_jtyp, 2, batch, row, col)

            # print("jmap in forward pass", jmap.shape)
            # print("lmap in forward pass",lmap.shape)

            if stack == 0:
                result["preds"] = {
                    "jmap": jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],
                    "lmap": lmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],
                    "joff": joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5,
                }
                if input_dict["mode"] == "testing":
                    return result

            L = OrderedDict()
            L["jmap"] = sum(
                weighted_cross_entropy_loss(jmap[i], T["jmap"][i]) for i in range(n_jtyp)
            )
            L["lmap"] = sum(
                cross_entropy_loss(lmap[i], T["lmap"][i]) for i in range(n_ltyp)
            )

            L["joff"] = sum(
                sigmoid_l1_loss(joff[i, j], T["joff"][i, j], -0.5, T["jmap"][i])
                for i in range(n_jtyp)
                for j in range(2)
            )
            for loss_name in L:
                L[loss_name].mul_(loss_weight[loss_name])
            losses.append(L)

            # for key, value in T.items():
            #     print(f"{key} shape when in results forward: {value.shape}")

        result["losses"] = losses
        return result


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    nlogp = -F.log_softmax(logits, dim=0)
    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


def weighted_cross_entropy_loss(logits, positive):
    # Calculate class frequencies
    positive_pixels = positive.sum()
    total_pixels = positive.numel()
    negative_pixels = total_pixels - positive_pixels

    # Calculate class weights
    w_1 = positive_pixels / total_pixels
    w_0 = negative_pixels / total_pixels

    w_1 = w_1 / (w_0+w_1)
    w_0 = w_0 / (w_0+w_1)

    # Compute weighted cross entropy loss
    nlogp = -F.log_softmax(logits, dim=0)
    loss = w_1 * positive * nlogp[1] + w_0 * (1 - positive) * nlogp[0]

    return loss.mean(2).mean(1)

def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)
