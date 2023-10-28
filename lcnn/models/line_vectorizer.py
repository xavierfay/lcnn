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
                del jc
            else:
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

        if input_dict["mode"] != "training":
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
            b = (cond2 | cond3 | cond4 ) & cond1
            lines = []
            score = []
            jcs_list = []
            jtype_list = []

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
                    loss_per_class[c] = loss_c.sum() * class_weights[
                        c]  # Summing up the loss and adjusting by class weight

                return loss_per_class

            class_weights = torch.tensor([1, 10, 10, 10]).to(x.device)

            y = torch.cat(ys)
            y = torch.argmax(y, dim=1)

            loss_per_class = cross_entropy_loss_per_class(x, y, class_weights)

            lneg = loss_per_class[0]
            lpos0 = loss_per_class[1]
            lpos1 = loss_per_class[2]
            lpos2 = loss_per_class[3]

            result["losses"][0]["lneg"] = lneg * M.loss_weight["lneg"]
            result["losses"][0]["lpos0"] = lpos0 * M.loss_weight["lpos0"]
            result["losses"][0]["lpos1"] = lpos1 * M.loss_weight["lpos1"]
            result["losses"][0]["lpos2"] = lpos2 * M.loss_weight["lpos2"]

        if input_dict["mode"] == "training":
            del result["preds"]

        # print(input_dict["mode"])
        # print("lines result:", len(lines))#, torch.max(lines))
        # print("results", len(result["preds"]["lines"][0].cpu().numpy()))
        # non_zero_count = torch.count_nonzero(result["preds"]["lines"][0].cpu())
        #
        # print("number of non zeros in tensor", non_zero_count.item() )

        return result

    def sample_lines(self, meta, jmap, joff, mode):
        with torch.no_grad():
            junc = meta["junc"]  # [N, 2]
            jtyp = meta["jtyp"]  # [N]
            Lpos = meta["Lpos"]  # [N+1, N+1]
            Lneg = meta["Lneg"]  # [N+1, N+1]
            lpre_label = meta["lpre_label"]  # [N, 3]

            n_type = jmap.shape[0]
            N = len(junc)
            device = jmap.device
            jmap = non_maximum_suppression(jmap).reshape(n_type, -1)
            joff = joff.reshape(n_type, 2, -1)
            #max_K = M.n_dyn_junc // n_type
            K_values = [150, 150] + [15] * 32
            assert len(K_values) == n_type
            scores = []
            indices = []
            updated_K_values = []
            for i in range(n_type):
                # Current layer's maximum allowable K
                current_max_K = K_values[i]

                # Calculate the number of values above the threshold for the current layer
                # above_threshold = (jmap[i] > M.eval_junc_thres).float().sum().item()
                #
                # if mode != "training":
                #     K = min(int(above_threshold), current_max_K)
                if mode != "training":
                    K = current_max_K
                else:
                    K = min(int(N * 2 + 2), current_max_K)

                # Ensure a minimum value for K
                if K < 2:
                    K = 2

                updated_K_values.append(K)
                # Get top K values and their indices for the current layer
                score, index = torch.topk(jmap[i], k=K)
                scores.append(score)
                indices.append(index)

            while len(updated_K_values) < len(K_values):
                updated_K_values.append(K_values[len(updated_K_values)])

            # Convert updated_K_values to the same type as K_values (assuming K_values is a list of integers)
            K_values = [int(k) for k in updated_K_values]
            # print(K_values)

            max_size = max([s.size(0) for s in scores])

            # Pad each tensor in scores and indices lists to match the max_size
            padded_scores = [F.pad(s, (0, max_size - s.size(0))) for s in scores]
            padded_indices = [F.pad(idx, (0, max_size - idx.size(0))) for idx in indices]

            # Convert lists to tensors for further processing
            score = torch.stack(padded_scores)
            index = torch.stack(padded_indices)

            y = (index // 256).float() + torch.gather(joff[:, 0], 1, index) + 0.5
            x = (index % 256).float() + torch.gather(joff[:, 1], 1, index) + 0.5

            # xy: [N_TYPE, K, 2]
            xy = torch.cat([y[..., None], x[..., None]], dim=-1)
            xy_ = xy[..., None, :]
            del x, y, index

            # dist: [N_TYPE, K, N]
            dist = torch.sum((xy_ - junc) ** 2, -1)
            cost, match = torch.min(dist, -1)

            for t in range(n_type):
                match[t, jtyp[match[t]] != t] = N
            match[cost > 1.5 * 1.5] = N
            match = match.flatten()

            if mode == "testing":
                match = (match - 1).clamp(min=0)

            # paring matrix:
            pairing_matrix = np.ones((n_type, n_type), dtype=int)
            # Modify the matrix based on the described pattern
            pairing_matrix[0, :1] = 0
            pairing_matrix[1, :2] = 0

            u, v = [], []
            for i in range(n_type):
                for j in range(n_type):
                    if pairing_matrix[i][j]:  # Check if the pairing is allowed
                        u_i, v_i = torch.meshgrid(
                            torch.arange(i * K_values[i], (i + 1) * K_values[i]),
                            torch.arange(j * K_values[j], (j + 1) * K_values[j])
                        )
                        u.append(u_i.flatten())
                        v.append(v_i.flatten())

            u = [ui.to(device) for ui in u]
            v = [vi.to(device) for vi in v]
            u, v = torch.cat(u).to(device), torch.cat(v).to(device)
            up, vp = match[u].to(device), match[v].to(device)

            scalar_labels = Lpos[up, vp]
            scalar_labels = scalar_labels.to(device).long()

            # Initialize a tensor of zeros with shape [N, 3]
            if mode == "training":
                c = torch.zeros_like(scalar_labels, dtype=torch.bool)

                # Sample negative Lines (Class 0)
                cdx = Lneg[up, vp].nonzero().flatten()
                if len(cdx) > M.n_dyn_negl:
                    # print("too many negative lines")
                    perm = torch.randperm(len(cdx), device=device)[: M.n_dyn_negl]
                    cdx = cdx[perm]
                c[cdx] = 1

                # Sample dashed Lines (Class 1)
                cdx = (scalar_labels == 1).nonzero().flatten()
                if len(cdx) > M.n_dyn_posl0:
                    perm = torch.randperm(len(cdx), device=device)[: M.n_dyn_posl0]
                    cdx = cdx[perm]
                c[cdx] = 1

                # Sample continous Lines (Class 2)
                cdx = (scalar_labels == 2).nonzero().flatten()
                if len(cdx) > M.n_dyn_posl1:
                    perm = torch.randperm(len(cdx), device=device)[: M.n_dyn_posl1]
                    cdx = cdx[perm]
                c[cdx] = 1

                # Sample connection Lines (Class 3)
                cdx = (scalar_labels == 3).nonzero().flatten()
                if len(cdx) > M.n_dyn_posl1:
                    perm = torch.randperm(len(cdx), device=device)[: M.n_dyn_posl2]
                    cdx = cdx[perm]
                c[cdx] = 1

                # sample other (unmatched) lines
                cdx = torch.randint(len(c), (M.n_dyn_othr,), device=device)
                c[cdx] = 1

            else:
                c = (u < v).flatten()

            if mode == "training":
                c = c.to(device).bool()

            #sample lines
            u, v, scalar_labels = u[c], v[c], scalar_labels[c]
            # print("u.shape:", u.shape, "v.shape:", v.shape)
            # print("Max value in u:", u.max().item())
            # print("Max value in v:", v.max().item())
            reshaped_xy = []
            for i in range(n_type):
                reshaped_xy.append(xy[i, :K_values[i]])
            xy = torch.cat(reshaped_xy, dim=0).to(device)
            # print("xy.shape:", xy.shape)

            xyu, xyv = xy[u].to(device), xy[v].to(device)

            label = torch.zeros(scalar_labels.shape[0], 4, device=device)
            # Assign a "1" in the respective column according to the scalar label
            label[torch.arange(label.shape[0]), scalar_labels] = 1
            line = torch.cat([xyu[:, None], xyv[:, None]], 1)

            jcs_list = []
            jtype_list = []

            xy_splits = torch.split(xy, K_values, dim=0)

            for i, xy_i in enumerate(xy_splits):
                score_i = score[i, :K_values[i]]
                valid_indices = score_i > 0.03
                subset = xy_i[valid_indices]

                if len(subset) > 0:  # Only append/extend when subset is non-empty
                    jcs_list.append(subset)
                    jtype_list.extend([i + 1] * len(subset))

            # Create flattened jcs tensor and jtype tensor
            jcs = torch.cat(jcs_list, dim=0)
            jtype = torch.tensor(jtype_list, device=xy.device)


            return line, label, jcs, jtype


# def non_maximum_suppression(a):
#     a = a.view(a.shape[0], 1, 256, 256)  # Reshape it to [n_type, 1, 256, 256]
#     ap = F.max_pool2d(a, 3, stride=1, padding=1)
#     keep = (a == ap).float()
#     a = a.view(a.shape[0], -1)  # Flatten it back after processing
#     return a * keep.view(keep.shape[0], -1)
def non_maximum_suppression(a):
    original_shape = a.shape
    # Reshape tensor to [1, n_type, 256, 256]
    a = a.view(1, original_shape[0], original_shape[1], original_shape[2])
    # Apply 3D max pooling across the layers and spatial dimensions
    ap = F.max_pool3d(a, (original_shape[0], 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))
    keep = (a == ap).float()
    a = a.view(original_shape[0], -1)  # Flatten it back after processing
    return a * keep.view(original_shape[0], -1)


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