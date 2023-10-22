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
        for task in ["jmap"]:
            print("jmap shape before permute: ", T[task].shape)
            T[task] = T[task].permute(1, 0, 2, 3)
        for task in ["joff"]:
            print("joff shape before permute: ", T[task].shape)
            T[task] = T[task].permute(1, 2, 0, 3, 4)

        offset = self.head_off
        loss_weight = M.loss_weight
        losses = []
        for stack, output in enumerate(outputs):
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()

            # Adjusting the reshape operations
            jmap = output[0: offset[0]].reshape(batch, 1, row, col)
            lmap = output[offset[0]: offset[1]].reshape(batch, n_ltyp, 1, row, col)  # keeping the extra dimension for lmap
            joff = output[offset[1]: offset[2]].reshape(2, batch, 1, row, col)

            if stack == 0:
                result["preds"] = {
                    "jmap": jmap.softmax(1).squeeze(1),  # removing the n_jtyp dimension after softmax
                    "lmap": lmap.permute(0, 1, 2, 3, 4).softmax(2).squeeze(2),
                    "joff": joff.permute(1, 0, 2, 3, 4).sigmoid().squeeze(2) - 0.5,
                }
                if input_dict["mode"] == "testing":
                    return result

            L = OrderedDict()
            L["jmap"] = cross_entropy_loss(jmap, T["jmap"])

            jmap_losses_per_class = cross_entropy_loss_per_class(jmap, T["jmap"])
            for idx, loss in enumerate(jmap_losses_per_class):
                L[f"jmap_class_{idx}"] = loss

            L["lmap"] = cross_entropy_loss(lmap.squeeze(2), T["lmap"])

            L["joff"] = sum(
                sigmoid_l1_loss(joff[j].squeeze(2), T["joff"][j], -0.5, T["jmap"])
                for j in range(2)
            )
            weights = [0.5] + [10.0] * (n_jtyp - 1)
            for idx, weight in enumerate(weights):
                loss_weight[f'jmap_class_{idx}'] = weight

            for loss_name in L:
                L[loss_name] = L[loss_name] * loss_weight[loss_name]
            losses.append(L)

            # for key, value in L.items():
            #     print(f"{key} shape when in results forward: {value}")

        result["losses"] = losses
        return result


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, target):
    # Ensure that target and logits have the same number of dimensions
    if len(logits.shape) != len(target.shape):
        target = target.unsqueeze(2)

    nlogp = F.log_softmax(logits, dim=1)
    return F.nll_loss(nlogp, target.long(), reduction='mean')

def cross_entropy_loss_per_class(logits, target):
    # Ensure that target and logits have the same number of dimensions
    if len(logits.shape) != len(target.shape):
        target = target.unsqueeze(2)

    # Get the softmax along the class dimension
    probs = F.softmax(logits, dim=1)

    # Convert target to one-hot encoding with casting to int64
    target_one_hot = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1).long(), 1)

    # Compute the loss for each class
    losses = -target_one_hot * torch.log(probs + 1e-8)

    # Return the mean loss for each class
    return losses.mean(dim=(0, 2, 3, 4))

def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)
