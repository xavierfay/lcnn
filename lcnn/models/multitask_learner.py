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
            jmap = jmap.permute(2, 0, 1, 3, 4)
            #jmap = nms_3d(jmap.softmax(0))

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

            alpha = compute_alpha(T["jmap"])
            # L["jmap"] = sum(
            #     combined_loss(jmap[i], T["jmap"][i], alpha) for i in range(n_jtyp)
            # )
            print("jmap shape", jmap.shape)
            print("T[jmap] shape", T["jmap"].shape)
            L["jmap"] = multi_class_focal_loss(jmap, T["jmap"], alpha)
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

            # for key, value in L.items():
            #     print(f"{key} loss when in results forward: {value}")

        result["losses"] = losses
        return result

def multi_class_focal_loss(logits, labels, alpha, gamma=2.0):
    """
    Compute the multi-class focal loss.
    Args:
    - logits (torch.Tensor): raw logits, shape [batch_size, n_classes, H, W]
    - labels (torch.Tensor): ground truth labels, shape [batch_size, n_classes, H, W]
    - alpha (torch.Tensor): class weights, shape [n_classes]
    - gamma (float): focusing parameter
    Returns:
    - loss (torch.Tensor): scalar tensor representing the loss
    """
    probas = F.softmax(logits, dim=1)
    p_t = (labels * probas).sum(dim=1)

    alpha_t = alpha[None, :, None, None].expand_as(logits)
    epsilon = 1e-7
    focal_term = (1 - p_t) ** gamma
    cross_entropy_term = -labels * torch.log(probas + epsilon)

    loss = (alpha_t * focal_term * cross_entropy_term).sum(dim=1)
    return loss.mean()

def nms_3d(a):
    n_jtyp, two, batch, row, col = a.shape

    result = torch.zeros_like(a)
    for i in range(n_jtyp):
        for j in range(two):
            slice_3d = a[i, j].unsqueeze(0)  # Add a dimension for max_pool3d
            ap = F.max_pool3d(slice_3d, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
            keep = (slice_3d == ap).float()
            result[i, j] = (slice_3d * keep).squeeze(0)  # Remove the added dimension

    return result


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    nlogp = -F.log_softmax(logits, dim=0)
    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


def focal_loss(logits, positive, alpha, gamma=2.0):
    # Get the probability of the positive class
    probas = F.softmax(logits, dim=1)  # Assuming logits shape: [batch, 2, height, width]

    mask = (positive == 1).float()
    p_t = mask * probas[:, 1, :, :] + (1.0 - mask) * probas[:, 0, :, :]

    # Extend alpha to have the same shape as logits
    alpha_t = alpha[None, :, None, None].expand_as(logits)

    epsilon = 1e-7
    loss = -alpha_t * (1 - p_t) ** gamma * torch.log(p_t + epsilon)
    return loss.mean()

def combined_loss(preds, targets, alpha):
    """
    preds: Predictions tensor of shape [batch, n_jtyp, height, width]
    targets: Ground truth tensor of shape [batch, n_jtyp, height, width]
    alpha: alpha values computed from compute_alpha function
    """
    n_jtyp = preds.shape[1]

    # Step 1: Compute Individual Focal Losses
    focal_losses = [focal_loss(preds[:, i, :, :], targets[:, i, :, :], alpha[i]) for i in range(n_jtyp)]
    individual_loss = sum(focal_losses)

    # Step 2: Enforce Exclusive Class Assignment
    max_vals, _ = preds.max(dim=1, keepdim=True)  # Find the maximum value among heatmaps for each pixel
    exclusive_preds = preds.clone() - max_vals  # Use clone() to avoid in-place operations

    # Step 3: Softmax Normalization
    normalized_preds = F.softmax(exclusive_preds, dim=1)

    # Combine the losses
    # Here, I'm combining the individual focal loss with a possible loss based on normalized_preds.
    # You might want to define what this second loss is. For now, I'm returning just the individual loss.
    return individual_loss  # or combine with another loss if desired

def compute_alpha(labels):
    """
    Compute the frequency of each class in the dataset.
    Args:
    - labels (torch.Tensor): a tensor of shape [batch_size, n_classes, H, W]
    Returns:
    - alpha (torch.Tensor): a tensor of shape [n_classes]
    """
    # Count the number of positive activations for each class
    class_counts = labels.sum(dim=(0, 2, 3))
    # Compute the frequency
    total_counts = labels.numel() / labels.shape[1]
    class_frequencies = class_counts / total_counts

    alpha = 1.0 / (class_frequencies + 1e-6)  # Adding a small constant to avoid division by zero
    return alpha


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



