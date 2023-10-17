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
        n_jtyp = int(torch.max(T["jmap"]).item())+1
        n_ltyp = int(torch.max(T["lmap"]).item())+1


        # switch to CNHW
        # for task in ["jmap"]:
        #     T[task] = T[task].permute(1, 0, 2, 3)
        for task in ["joff"]:
            T[task] = T[task].permute(1, 0, 2, 3)

        offset = self.head_off
        loss_weight = M.loss_weight
        losses = []
        for stack, output in enumerate(outputs):
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()
            # jmap now predicts class scores for each pixel
            jmap = output[0: offset[0]].reshape(batch, n_jtyp, row, col)
            lmap = output[offset[0]: offset[1]].reshape(batch, n_ltyp, row, col)
            joff = output[offset[1]: offset[2]].reshape(2, batch, row, col)

            if stack == 0:
                result["preds"] = {
                    "jmap": jmap.softmax(1),
                    "lmap": lmap.softmax(1),
                    "joff": joff.permute(1, 0, 2, 3).sigmoid() - 0.5,
                }
                if input_dict["mode"] == "testing":
                    return result

            L = OrderedDict()
            L["jmap"] = cross_entropy_loss(jmap, T["jmap"])

            #print(lmap.shape, T["lmap"].shape)
            L["lmap"] = cross_entropy_loss(lmap, T["lmap"])

            #print(joff.shape, T["joff"].shape)
            L["joff"] = sum(
                sigmoid_l1_loss(joff[j], T["joff"][j], -0.5, T["jmap"])
                for j in range(2)
            )

            for loss_name in L:
                L[loss_name].mul_(loss_weight[loss_name])
            losses.append(L)

            # for key, value in L.items():
            #     print(f"{key} shape when in results forward: {value}")

        result["losses"] = losses
        return result


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, target):
    # print(target.max())
    # print(logits.size(1))
    # print(target.min())
    nlogp = F.log_softmax(logits, dim=1)
    return F.nll_loss(nlogp, target.long(), reduction='mean')


def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)
