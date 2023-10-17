import itertools
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lcnn.config import M

FEATURE_DIM = 0


class LineVectorizer(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        lambda_ = torch.linspace(0, 1, M.n_pts0)[:, None]
        self.register_buffer("lambda_", lambda_)
        self.do_static_sampling = M.n_stc_posl + M.n_stc_negl > 0

        self.fc1 = nn.Conv2d(256, M.dim_loi, 1)
        scale_factor = M.n_pts0 // M.n_pts1
        if M.use_conv:
            self.pooling = nn.Sequential(
                nn.MaxPool1d(scale_factor, scale_factor),
                Bottleneck1D(M.dim_loi, M.dim_loi),
            )
            self.fc2 = nn.Sequential(
                nn.ReLU(inplace=True), nn.Linear(M.dim_loi * M.n_pts1 + FEATURE_DIM, 3)
            )
        else:
            self.pooling = nn.MaxPool1d(scale_factor, scale_factor)
            self.fc2 = nn.Sequential(
                nn.Linear(M.dim_loi * M.n_pts1 + FEATURE_DIM, M.dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(M.dim_fc, M.dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(M.dim_fc, 3),
            )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_dict):
        result = self.backbone(input_dict)
        h = result["preds"]
        x = self.fc1(result["feature"])
        n_batch, n_channel, row, col = x.shape

        xs, ys, fs, ps, idx, jcs = [], [], [], [], [0], []
        for i, meta in enumerate(input_dict["meta"]):
            p, label, jc = self.sample_lines(
                meta, h["jmap"][i], h["joff"][i], input_dict["mode"]
            )
            # print("p.shape:", p.shape)
            ys.append(label)
            if input_dict["mode"] == "training" and self.do_static_sampling:
                p = torch.cat([p, meta["lpre"]])
                ys.append(meta["lpre_label"])
                del jc
            else:
                jcs.append(jc)
                ps.append(p)


            p = p[:, 0:1, :] * self.lambda_ + p[:, 1:2, :] * (1 - self.lambda_) - 0.5
            p = p.reshape(-1, 2)  # [N_LINE x N_POINT, 2_XY]
            px, py = p[:, 0].contiguous(), p[:, 1].contiguous()
            px0 = px.floor().clamp(min=0, max=255)
            py0 = py.floor().clamp(min=0, max=255)
            px1 = (px0 + 1).clamp(min=0, max=255)
            py1 = (py0 + 1).clamp(min=0, max=255)
            px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

            # xp: [N_LINE, N_CHANNEL, N_POINT]
            xp = (
                (
                    x[i, :, px0l, py0l] * (px1 - px) * (py1 - py)
                    + x[i, :, px1l, py0l] * (px - px0) * (py1 - py)
                    + x[i, :, px0l, py1l] * (px1 - px) * (py - py0)
                    + x[i, :, px1l, py1l] * (px - px0) * (py - py0)
                )
                .reshape(n_channel, -1, M.n_pts0)
                .permute(1, 0, 2)
            )
            xp = self.pooling(xp)
            xs.append(xp)
            idx.append(idx[-1] + xp.shape[0])

        x, y = torch.cat(xs), torch.cat(ys)
        x = x.reshape(-1, M.n_pts1 * M.dim_loi)
        x = self.fc2(x)

        if input_dict["mode"] != "training":
            p = torch.cat(ps)
            s = torch.softmax(x, -1)
            cond1 = s[:, 0] > 0.5
            cond2 = s[:, 1] > 0.9
            cond3 = s[:, 2] > 0.3

            # Combine the conditions using logical OR
            b = cond1 | cond2 | cond3
            #b = (s > 0.3).any(dim=-1)
            lines = []
            score = []
            for i in range(n_batch):
                p0 = p[idx[i] : idx[i + 1]]
                s0 = s[idx[i] : idx[i + 1]]
                mask = b[idx[i] : idx[i + 1]]
                p0 = p0[mask]
                s0 = s0[mask]
                if len(p0) == 0:
                    lines.append(torch.zeros([1, M.n_out_line, 2, 2], device=p.device))
                    score.append(torch.zeros([1, M.n_out_line, 3], device=p.device))
                else:
                    max_score_indices = torch.argmax(s0, dim=1)
                    arg = torch.argsort(max_score_indices, descending=True)
                    p0, s0 = p0[arg], s0[arg]
                    lines.append(p0[None, torch.arange(M.n_out_line) % len(p0)])
                    score.append(s0[None, torch.arange(M.n_out_line) % len(s0)])
                for j in range(len(jcs[i])):
                    if len(jcs[i][j]) == 0:
                        jcs[i][j] = torch.zeros([M.n_out_junc, 2], device=p.device)
                    jcs[i][j] = jcs[i][j][
                        None, torch.arange(M.n_out_junc) % len(jcs[i][j])
                    ]
            result["preds"]["lines"] = torch.cat(lines)
            result["preds"]["score"] = torch.cat(score)
            result["preds"]["juncs"] = torch.cat([jcs[i][0] for i in range(n_batch)])
            if len(jcs[i]) > 1:
                result["preds"]["junts"] = torch.cat(
                    [jcs[i][1] for i in range(n_batch)]
                )

        if input_dict["mode"] != "testing":


            def cross_entropy_loss_per_class(x, y, class_weights, num_classes=3):
                # Ensure the logits are float, Convert labels to long
                x = x.float()
                y = y.long()

                # Calculate the softmax along the second dimension
                softmax = torch.exp(x) / torch.exp(x).sum(dim=-1, keepdim=True)

                # Initialize an empty tensor to store the per-class losses
                loss_per_class = torch.zeros(num_classes).float().to(
                    x.device)  # ensure the tensor is on the same device as x

                # Loop over each class and calculate the loss
                for c in range(num_classes):
                    # Create a mask that selects only the samples of class c
                    mask = (y == c).float()
                    loss_c = -torch.log(softmax[:, c] + 1e-8) * mask  # adding a small value to avoid log(0)
                    loss_per_class[c] = loss_c.sum() * class_weights[
                        c]  # Summing up the loss and adjusting by class weight

                # Normalize by the total number of samples in the batch
                loss_per_class /= x.shape[0]

                return loss_per_class

            class_weights = torch.tensor([100, 100, 100]).to(x.device)

            y = torch.argmax(y, dim=1)
            count = torch.bincount(y)
            unique_values = torch.unique(y)
            print(unique_values, count)
            loss_per_class = cross_entropy_loss_per_class(x, y, class_weights)

            lneg = loss_per_class[0]
            lpos0 = loss_per_class[1]
            lpos1 = loss_per_class[2]


            result["losses"][0]["lneg"] = lneg * M.loss_weight["lneg"]
            result["losses"][0]["lpos0"] = lpos0 * M.loss_weight["lpos0"]
            result["losses"][0]["lpos1"] = lpos1 * M.loss_weight["lpos1"]


        if input_dict["mode"] == "training":
            del result["preds"]

        return result

    def sample_lines(self, meta, jmap, joff, mode):
        with torch.no_grad():
            junc = meta["junc"]  # [N, 2]
            jtyp = meta["jtyp"]  # [N]
            Lpos = meta["Lpos"]  # [N+1, N+1]
            Lneg = meta["Lneg"]  # [N+1, N+1]

            n_type = jmap.shape[0]
            jmap = non_maximum_suppression(jmap).reshape(n_type, -1)
            joff = joff.reshape(n_type, 2, -1)
            max_K = M.n_dyn_junc // n_type
            N = len(junc)
            if mode != "training":
                K = min(int((jmap > M.eval_junc_thres).float().sum().item()), max_K)
            else:
                K = min(int(N * 2 + 2), max_K)
            if K < 2:
                K = 2
            device = jmap.device

            # # index: [N_TYPE, K]
            # score, index = torch.topk(jmap, k=K)
            # y = (index // 256).float() + torch.gather(joff[:, 0], 1, index) + 0.5
            # x = (index % 256).float() + torch.gather(joff[:, 1], 1, index) + 0.5
            #
            # # xy: [N_TYPE, K, 2]
            # xy = torch.cat([y[..., None], x[..., None]], dim=-1)
            # SCORE_THRESHOLD = 0.1
            # mask = score > SCORE_THRESHOLD  # mask: [N_TYPE, K]
            #
            # # Find indices where mask is True
            # true_indices = torch.nonzero(mask, as_tuple=True)
            #
            # # Apply mask to xy using true_indices
            # xy = xy[true_indices]
            #
            # # Subsequent operations...
            # xy_ = xy[..., None, :]
            # del x, y, index
            #
            # # Compute distances and find matches
            # dist = torch.sum((xy_ - junc) ** 2, -1)
            # cost, match = torch.min(dist, -1)
            #
            # # Convert the indices into a 1D tensor
            # flattened_indices = true_indices[0] * mask.size(1) + true_indices[1]
            #
            # # Apply mask to match using true_indices
            # match = match[flattened_indices]
            #
            #
            # # Filter or modify match based on some conditions
            # for t in range(n_type):
            #     match[t, jtyp[match[t]] != t] = N
            #     print("t:", t)
            #     print("match[t]:", match[t])
            #     print("jtyp.size():", jtyp.size())
            #     print("match.size():", match.size())
            #
            # match[cost > 1.5 * 1.5] = N
            #
            # # if mode == "testing":
            # #     match = (match - 1).clamp(min=0)
            #
            #
            # _ = torch.arange(n_type * K, device=device)
            # u, v = torch.meshgrid(_, _)
            # u, v = u.flatten(), v.flatten()
            # up, vp = match[u], match[v]

            threshold = 0.03  # Define your threshold value here

            score, index = torch.topk(jmap, k=K)

            # Create a mask for scores that are higher than the threshold
            mask = score > threshold

            # Filter out indices, scores, joff, and junc based on the mask
            filtered_score = score[mask]
            filtered_index = index[mask]
            # Expand the filtered_index to match the batch size and add an extra dimension at the end
            # Adjust the shape of filtered_index
            print(junc.shape)
            print(filtered_index.shape)
            expanded_index = filtered_index.unsqueeze(0).unsqueeze(1).expand(2, 2, -1)

            # Gather values from joff along the third dimension
            filtered_joff = torch.gather(joff, 2, expanded_index)

            # Adjust the shape of filtered_index for junc
            expanded_index_for_junc = filtered_index.unsqueeze(-1)

            # Gather values from junc along the first dimension
            filtered_junc = torch.gather(junc, 0, expanded_index_for_junc)

            # Calculate x, y based on the filtered indices
            y = (filtered_index // 128).float() + torch.gather(filtered_joff[:, 0], 1, filtered_index) + 0.5
            x = (filtered_index % 128).float() + torch.gather(filtered_joff[:, 1], 1, filtered_index) + 0.5

            # xy: [N_TYPE, K, 2]
            xy = torch.cat([y[..., None], x[..., None]], dim=-1)
            xy_ = xy[..., None, :]
            del x, y, filtered_index

            # dist: [N_TYPE, K, N]
            dist = torch.sum((xy_ - filtered_junc) ** 2, -1)
            cost, match = torch.min(dist, -1)

            # xy: [N_TYPE * K, 2]
            # match: [N_TYPE, K]
            for t in range(n_type):
                match[t, jtyp[match[t]] != t] = N
            match[cost > 1.5 * 1.5] = N
            match = match.flatten()

            _ = torch.arange(n_type * K, device=device)
            u, v = torch.meshgrid(_, _)
            u, v = u.flatten(), v.flatten()
            up, vp = match[u], match[v]

            scalar_labels = Lpos[up, vp]
            scalar_labels = scalar_labels.long()
            # Initialize a tensor of zeros with shape [N, 3]
            label = torch.zeros(scalar_labels.shape[0], 3, device=scalar_labels.device)

            # Assign a "1" in the respective column according to the scalar label
            label[torch.arange(label.shape[0]), scalar_labels] = 1

            if mode == "training":
                c = torch.zeros_like(label[:, 0], dtype=torch.bool)

                # Sample negative Lines (Class 0)
                cdx = Lneg[up, vp].nonzero().flatten()
                if len(cdx) > M.n_dyn_negl:
                    # print("too many negative lines")
                    perm = torch.randperm(len(cdx), device=device)[: M.n_dyn_negl]
                    cdx = cdx[perm]
                c[cdx] = 1

                # Sample dashed Lines (Class 1)
                cdx = (label[:, 1] == 1).nonzero().flatten()
                if len(cdx) > M.n_dyn_negl:
                    perm = torch.randperm(len(cdx), device=device)[: M.n_dyn_posl0]
                    cdx = cdx[perm]
                c[cdx] = 1

                # Sample continous Lines (Class 2)
                cdx = (label[:, 2] == 1).nonzero().flatten()
                if len(cdx) > M.n_dyn_othr:
                    perm = torch.randperm(len(cdx), device=device)[: M.n_dyn_posl1]
                    cdx = cdx[perm]
                c[cdx] = 1

                # sample other (unmatched) lines
                cdx = torch.randint(len(c), (M.n_dyn_othr,), device=device)
                c[cdx] = 1

            else:
                c = (u < v).flatten()

            # sample lines
            u, v, label = u[c], v[c], label[c]
            xy = xy.reshape(n_type * K, 2)
            xyu, xyv = xy[u], xy[v]

            # Compute slopes and create masks for valid lines (horizontal/vertical)
            deltas = xyv - xyu
            slopes = torch.where(deltas[:, 0] != 0, deltas[:, 1] / deltas[:, 0], float('inf'))
            horizontal_mask = torch.abs(slopes) < 0.01
            vertical_mask = torch.abs(slopes) > 1000
            valid_lines_mask = horizontal_mask | vertical_mask
            # print("shapes", valid_lines_mask.shape[0], xyu.shape[0])


            # Filter xyu, xyv, and label using the valid_lines_mask
            xyu, xyv = xyu[valid_lines_mask], xyv[valid_lines_mask]
            label = label[valid_lines_mask]

            line = torch.cat([xyu[:, None], xyv[:, None]], 1)
            xy = xy.reshape(n_type, K, 2)
            # jcs = [xy[i, score[i].long()] for i in range(n_type)]
            jcs = [xy[i, score[i] > 0.03] for i in range(n_type)]

            return line, label, jcs

def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask


class Bottleneck1D(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Bottleneck1D, self).__init__()

        planes = outplanes // 2
        self.op = nn.Sequential(
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv1d(inplanes, planes, kernel_size=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, outplanes, kernel_size=1),
        )

    def forward(self, x):
        return x + self.op(x)