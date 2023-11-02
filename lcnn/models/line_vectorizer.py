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
        self.do_static_sampling = M.n_stc_posl0 + M.n_stc_posl1 + M.n_stc_posl2 + M.n_stc_negl > 0

        self.fc1 = nn.Conv2d(256, M.dim_loi, 1)
        scale_factor = M.n_pts0 // M.n_pts1
        if M.use_conv:
            self.pooling = nn.Sequential(
                nn.MaxPool1d(scale_factor, scale_factor),
                Bottleneck1D(M.dim_loi, M.dim_loi),
            )
            self.fc2 = nn.Sequential(
                nn.ReLU(inplace=True), nn.Linear(M.dim_loi * M.n_pts1 + FEATURE_DIM, 4)
            )
        else:
            self.pooling = nn.MaxPool1d(scale_factor, scale_factor)
            self.fc2 = nn.Sequential(
                nn.Linear(M.dim_loi * M.n_pts1 + FEATURE_DIM, M.dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(M.dim_fc, M.dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(M.dim_fc, 4),
            )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_dict):
        result = self.backbone(input_dict)
        h = result["preds"]
        x = self.fc1(result["feature"])
        n_batch, n_channel, row, col = x.shape


        xs, ys, fs, ps, idx, jcs, jtypes = [], [], [], [], [0], [], []
        for i, meta in enumerate(input_dict["meta"]):

            p, label, jc, jtype = self.sample_lines(
                meta, h["jmap"][i], h["joff"][i], input_dict["mode"]
            )
            # print("p.shape:", p.shape)
            ys.append(label)
            if input_dict["mode"] == "training" and self.do_static_sampling:
                p = torch.cat([p, meta["lpre"]])
                #feat = torch.cat([feat, meta["lpre_feat"]])
                ys.append(meta["lpre_label"])
                #del jc

            jcs.append(jc)
            ps.append(p)
            jtypes.append(jtype)
            #fs.append(feat)


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

        x= torch.cat(xs)
        y = torch.cat(ys)
        x = x.reshape(-1, M.n_pts1 * M.dim_loi)
        if M.use_half:
            x = x.half()
        x = self.fc2(x)

        #if input_dict["mode"] != "training":
        p = torch.cat(ps)
        s = torch.softmax(x, -1)
        cond1 = s[:, 0] < 0.25
        cond2 = s[:, 1] > 0.25
        cond3 = s[:, 2] > 0.25
        cond4 = s[:, 3] > 0.25

        # s_arg = torch.argmax(s, dim=1)
        #
        # cond1 = s_arg != 0
        # cond2 = s[:, 1] > 0.1
        # cond3 = s[:, 2] > 0.1
        # cond4 = s[:, 3] > 0.1

        # Combine the conditions using logical OR
        b = (cond2 | cond3 | cond4) & cond1
        lines = []
        score = []

        for i in range(n_batch):
            p0 = p[idx[i]: idx[i + 1]]
            s0 = s[idx[i]: idx[i + 1]]
            mask = b[idx[i]: idx[i + 1]]
            p0 = p0[mask]
            s0 = s0[mask]
            if len(p0) == 0:
                lines.append(torch.zeros([1, M.n_out_line, 2, 2], device=p.device))
                score.append(torch.zeros([1, M.n_out_line, 4], device=p.device))
            else:
                max_score_indices = torch.argmax(s0, dim=1)
                arg = torch.argsort(max_score_indices, descending=True)
                p0, s0 = p0[arg], s0[arg]
                #print("shape p0", p0.shape)
                lines.append(p0[None, torch.arange(M.n_out_line) % len(p0)])
                score.append(s0[None, torch.arange(M.n_out_line) % len(s0)])
            if len(jcs[i]) == 0:
                jcs[i] = torch.zeros([M.n_out_junc, 2], device=p.device)
                jtypes[i] = torch.zeros([M.n_out_junc], device=p.device)

            jcs[i] = jcs[i][
                None, torch.arange(M.n_out_junc) % len(jcs[i])
            ]
            jtypes[i] = jtypes[i][
                None, torch.arange(M.n_out_junc) % len(jtypes[i])
            ]

        result["preds"]["lines"] = torch.cat(lines)
        result["preds"]["score"] = torch.cat(score)
        result["preds"]["juncs"] = torch.cat([jcs[i] for i in range(n_batch)])
        result["preds"]["jtype"] = torch.cat([jtypes[i] for i in range(n_batch)])

            #print("length lines", result["preds"]["lines"].shape)




        if input_dict["mode"] != "testing":
            def cross_entropy_loss_per_class(x, y, class_weights, num_classes=4, misclass_penalty=10):
                # Ensure the logits are float, Convert labels to long
                x = x.float()
                y = y.long()

                # Calculate the log softmax along the second dimension
                log_softmax = F.log_softmax(x, dim=-1)

                # Initialize an empty tensor to store the per-class losses
                loss_per_class = torch.zeros(num_classes).float().to(
                    x.device)  # ensure the tensor is on the same device as x

                # Loop over each class and calculate the loss
                for c in range(num_classes):
                    # Create a mask that selects only the samples of class c
                    mask = (y == c).float()
                    loss_c = -log_softmax[:, c] * mask
                    loss_per_class[c] = loss_c.sum() * class_weights[c]

                # Normalize by the total number of samples in the batch
                misclass_mask = (y == 0).float() * (torch.argmax(loss_per_class, dim=0) == 1).float()
                misclass_loss = misclass_mask.sum() * misclass_penalty

                loss_per_class[0] += misclass_loss

                loss_per_class /= x.shape[0]
                return loss_per_class

            class_weights = torch.tensor([1, 10, 10, 10]).to(x.device)

            y = torch.argmax(y, dim=1)

            if input_dict["mode"] != "training":
                count = torch.bincount(y)
                unique_values = torch.unique(y)
                print("values of labels",unique_values, count)

                x_class = torch.argmax(x, dim=1)
                count = torch.bincount(x_class)
                unique_values = torch.unique(x_class)
                print("values of pred", unique_values, count)

            loss_per_class = cross_entropy_loss_per_class(x, y, class_weights)

            lneg = loss_per_class[0]
            lpos0 = loss_per_class[1]
            lpos1 = loss_per_class[2]
            lpos2 = loss_per_class[3]

            result["losses"][0]["lneg"] = lneg * M.loss_weight["lneg"]
            result["losses"][0]["lpos0"] = lpos0 * M.loss_weight["lpos0"]
            result["losses"][0]["lpos1"] = lpos1 * M.loss_weight["lpos1"]
            result["losses"][0]["lpos2"] = lpos2 * M.loss_weight["lpos2"]


        # print(input_dict["mode"])
        # print("lines result:", len(lines))#, torch.max(lines))
        # print("results", len(result["preds"]["lines"][0].cpu().numpy()))
        # non_zero_count = torch.count_nonzero(result["preds"]["lines"][0].cpu())
        #
        # print("number of non zeros in tensor", non_zero_count.item() )

        return result


    def sample_lines(self, meta, jmap, joff, mode):
        with torch.no_grad():
            junc, jtyp = meta["junc"], meta["jtyp"]
            Lpos, Lneg = meta["Lpos"], meta["Lneg"]
            device = jmap.device

            jmap = nms_3d(jmap)
            n_type = jmap.size(1)
            max_K = M.n_dyn_junc
            N = len(junc)
            if mode != "training":
                K = min(int((jmap > M.eval_junc_thres).float().sum().item()), max_K)
            else:
                K = min(int(N * 2 + 2), max_K)
            if K < 2:
                K = 2

            K = min(K, jmap.numel())

            def get_top_k_3d(jmap, joff, K, device):
                # Transfer tensors to the specified device
                jmap = jmap.to(device)
                joff = joff.to(device)

                # Flatten jmap and Get top K scores and their indices
                score, index = torch.topk(jmap.view(-1), k=K)

                # Get 3D coordinates
                depth = jmap.size(2)
                height = jmap.size(1)

                jtype = (index // (depth * height)).long()
                y = ((index % (depth * height)) // depth).float()
                x = (index % depth).float()

                # Adjust coordinates with offsets
                y += torch.gather(joff[:, 0].contiguous().view(-1), 0, index) + 0.5
                x += torch.gather(joff[:, 1].contiguous().view(-1), 0, index) + 0.5

                return score, jtype, x, y

            print("k", K)
            print("N", N)
            print("jmap shape:", jmap.shape)
            print("jmap size:", jmap.numel())
            score, jtype, x, y = get_top_k_3d(jmap, joff, K, device)
            print("score", score.shape)
            print("jtype", jtype.shape)

            # xy: [N_TYPE, K, 2]
            xy = torch.cat([y[..., None], x[..., None]], dim=-1)
            xy_ = xy[..., None, :]
            del x, y

            print("xy", xy.shape)

            # dist: [N_TYPE, K, N]
            dist = torch.sum((xy_ - junc) ** 2, -1)
            cost, match = torch.min(dist, -1)

            match[cost > 1.5 * 1.5] = N
            match = match.flatten()

            # Create mesh grid and filter based on conditions
            u, v = torch.meshgrid(torch.arange(K, device=device), torch.arange(K, device=device))
            u, v = u.flatten(), v.flatten()
            up, vp = match[u], match[v]
            scalar_labels = Lpos[up, vp].long()

            c = (u < v).flatten() if mode != "training" else self.sample_training_labels(scalar_labels, Lneg, up, vp,
                                                                                         device)

            # Create line and label tensors
            u, v, scalar_labels = u[c], v[c], scalar_labels[c]
            xy = xy.reshape(K, 2)
            line = torch.stack([xy[u], xy[v]], dim=1)
            label = torch.zeros(scalar_labels.shape[0], 4, device=scalar_labels.device)
            label[torch.arange(label.shape[0]), scalar_labels] = 1

            # Process jcs and jtype
            jcs = xy[score > 0.001]
            jtype_tensor = jtype[score > 0.001]

            #jcs, jtype = self.matching_algorithm(xy, jmap, score)

            return line, label, jcs, jtype_tensor



    def matching_algorithm(self, xy, jmap, score):
        n_type, K, _ = xy.shape
        xy_int = xy.long()
        if not (xy_int >= 0).all():
            print("Negative indices detected:", xy_int[xy_int < 0])

        if not (xy_int[:, :, 0] < jmap.shape[-1]).all():
            print("X indices out of bounds:", xy_int[:, :, 0][xy_int[:, :, 0] >= jmap.shape[-1]])

        if not (xy_int[:, :, 1] < jmap.shape[-2]).all():
            print("Y indices out of bounds:", xy_int[:, :, 1][xy_int[:, :, 1] >= jmap.shape[-2]])

        xy_int = xy_int.clamp(min=0, max=jmap.shape[-2] - 1)  # For Y-dimension
        xy_int[:, :, 0] = xy_int[:, :, 0].clamp(min=0, max=jmap.shape[-1] - 1)  # For X-dimension

        # The first two layers of xy always correspond to the first two layers of jmap
        jtype_0_1 = torch.arange(2, device=jmap.device).view(2, 1).expand(2, K)
        # For the third layer of xy, get its intensity across jmap[2:]
        intensities_2 = jmap[2:, xy_int[2, :, 1], xy_int[2, :, 0]]
        # Determine the jtype for the third layer of xy based on the maximum intensity
        jtype_2 = torch.argmax(intensities_2, dim=0).add(2)  # Add 2 to offset for the first two layers
        # Combine jtypes
        jtype = torch.cat([jtype_0_1, jtype_2.unsqueeze(0)], dim=0)  # Shape: [3, 200]
        # Filter xy and jtype based on the score threshold
        valid_indices = score > 0.001
        jcs = xy[valid_indices]

        filtered_jtype = jtype[valid_indices]

        return jcs, filtered_jtype

    def sample_training_labels(self, scalar_labels, Lneg, up, vp, device):
        c = torch.zeros_like(scalar_labels, dtype=torch.bool)
        for class_idx, max_samples in enumerate([M.n_dyn_negl, M.n_dyn_posl0, M.n_dyn_posl1, M.n_dyn_posl2]):
            cdx = (scalar_labels == class_idx).nonzero().flatten()
            if len(cdx) > max_samples:
                cdx = torch.randperm(len(cdx), device=device)[:max_samples]
            c[cdx] = 1
        cdx = torch.randint(len(c), (M.n_dyn_othr,), device=device)
        c[cdx] = 1
        return c


def nms_3d(a):
    original_shape = a.shape

    # For multiple layers, apply 3D NMS
    a = a.view(1, original_shape[0], original_shape[1], original_shape[2])
    ap = F.max_pool3d(a, (original_shape[0], 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))
    keep = (a == ap).float()
    return (a * keep).squeeze(0)

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