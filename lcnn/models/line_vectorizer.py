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
                #feat = torch.cat([feat, meta["lpre_feat"]])
                ys.append(meta["lpre_label"])
                del jc
            else:
                jcs.append(jc)
                ps.append(p)
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

        x, y = torch.cat(xs), torch.cat(ys)
        #f = torch.cat(fs)
        x = x.reshape(-1, M.n_pts1 * M.dim_loi)
        #x = torch.cat([x, f], 1)
        x = self.fc2(x)

        if input_dict["mode"] != "training":
            p = torch.cat(ps)
            s = torch.softmax(x, -1)
            b = (s > 0.01).any(dim=-1)
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
                    score.append(torch.zeros([1, M.n_out_line], device=p.device))
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
            y = torch.cat(ys)
            y = torch.argmax(y, dim=1) #.long()
            #y = y.float()
            #x = torch.softmax(x, dim=-1)
            #x = x.float()
            # print("this is x, y", x[1], y[1])
            loss = self.loss(x, y)
            #lpos_mask, lneg_mask = y, 2 - y
            lpos_dashed_mask = (y == 1).float()
            lpos_continous_mask = (y == 2).float()
            lpos_mask = lpos_continous_mask + lpos_dashed_mask
            lneg_mask = (y == 0).float()
            loss_lpos, loss_lneg = loss * lpos_mask, loss * lneg_mask

            def sum_batch(x):
                xs = [x[idx[i] : idx[i + 1]].sum()[None] for i in range(n_batch)]
                return torch.cat(xs)

            lpos = sum_batch(loss_lpos) / sum_batch(lpos_mask).clamp(min=1)
            lneg = sum_batch(loss_lneg) / sum_batch(lneg_mask).clamp(min=1)
            result["losses"][0]["lpos"] = lpos * M.loss_weight["lpos"]
            result["losses"][0]["lneg"] = lneg * M.loss_weight["lneg"]

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

            # print("junc:", junc, junc.shape)
            # print("jtype", jtyp, jtyp.shape)
            # print("Lpos:", Lpos, Lpos.shape)
            # print("Lneg", Lneg, Lneg.shape)


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

            # index: [N_TYPE, K]
            score, index = torch.topk(jmap, k=K)
            y = (index // 256).float() + torch.gather(joff[:, 0], 1, index) + 0.5
            x = (index % 256).float() + torch.gather(joff[:, 1], 1, index) + 0.5

            # xy: [N_TYPE, K, 2]
            xy = torch.cat([y[..., None], x[..., None]], dim=-1)
            xy_ = xy[..., None, :]
            del x, y, index

            #print("xy_", xy_.shape, xy_)

            # dist: [N_TYPE, K, N]
            dist = torch.sum((xy_ - junc) ** 2, -1)
            cost, match = torch.min(dist, -1)

            # xy: [N_TYPE * K, 2]
            # match: [N_TYPE, K]
            # TODO: this flatten can help

            for t in range(n_type):
                match[t, jtyp[match[t]] != t] = N
            match[cost > 1.5 * 1.5] = N
            match = match.flatten()

            if mode == "testing":
                match = (match - 1).clamp(min=0)

            # class_two_indices = (Lpos == 2).nonzero(as_tuple=True)

            _ = torch.arange(n_type * K, device=device)
            u, v = torch.meshgrid(_, _)
            u, v = u.flatten(), v.flatten()
            up, vp = match[u], match[v]
            #print("up max",torch.max(up))

            # Ensuring Class 2 Inclusion in up and vp
            # Define how many entries you want to ensure are class 2
            # num_class_two_to_include = 100
            #
            # # Randomly select some class 2 indices
            # selected_indices = torch.randint(0, len(class_two_indices[0]), (num_class_two_to_include,))
            #
            # selected_class_two_indices_row = class_two_indices[0][selected_indices]
            # selected_class_two_indices_col = class_two_indices[1][selected_indices]
            #
            # # Replace some of the initially sampled indices with class 2 indices
            # up[:num_class_two_to_include] = selected_class_two_indices_row
            # vp[:num_class_two_to_include] = selected_class_two_indices_col

            # # Optionally shuffle up and vp if order matters
            # up = up[torch.randperm(up.size(0))]
            # vp = vp[torch.randperm(vp.size(0))]

            scalar_labels = Lpos[up, vp]
            scalar_labels = scalar_labels.long()
            # Initialize a tensor of zeros with shape [N, 3]
            label = torch.zeros(scalar_labels.shape[0], 3, device=scalar_labels.device)

            # Assign a "1" in the respective column according to the scalar label
            label[torch.arange(label.shape[0]), scalar_labels] = 1
            # print("after sampling", label, torch.max(label), label.shape)

            if mode == "training":
                c = torch.zeros_like(label[:, 0], dtype=torch.bool)

                # Sample negative Lines (Class 0)
                cdx = (label[:, 0] == 1).nonzero().flatten()
                if len(cdx) > M.n_dyn_posl:
                    perm = torch.randperm(len(cdx), device=device)[: M.n_dyn_posl]
                    cdx = cdx[perm]
                c[cdx] = 1

                # Sample continous Lines (Class 1)
                cdx = (label[:, 1] == 1).nonzero().flatten()
                if len(cdx) > M.n_dyn_negl:
                    perm = torch.randperm(len(cdx), device=device)[: M.n_dyn_negl]
                    cdx = cdx[perm]
                c[cdx] = 1

                # Sample dashed Lines (Class 2)
                cdx = (label[:, 2] == 1).nonzero().flatten()
                if len(cdx) > M.n_dyn_othr:
                    perm = torch.randperm(len(cdx), device=device)[: M.n_dyn_othr]
                    cdx = cdx[perm]
                c[cdx] = 1

            else:
                c = (u < v).flatten()

            #sample lines
            u, v, label = u[c], v[c], label[c]

            #print("label before straight line",label)
            xy = xy.reshape(n_type * K, 2)
            xy = xy.reshape(n_type, K, 2)
            jcs = [xy[i, score[i] > 0.03] for i in range(n_type)]
            jcs_flat = torch.cat(jcs, dim=0)
            xyu, xyv = jcs_flat[u], jcs_flat[v]

            deltas=xyv-xyu
            slopes=torch.where(deltas[:,0]!=0,deltas[:,1]/deltas[:,0],float('inf'))

            #maskforhorizontallines
            horizontal_mask=torch.abs(slopes)<0.05

            #maskforverticallines
            vertical_mask=torch.abs(slopes)>100 # A large number to approximate infinity

            valid_lines_mask=horizontal_mask|vertical_mask
            xyu,xyv = xyu[valid_lines_mask],xyv[valid_lines_mask]
            u,v = u[valid_lines_mask],v[valid_lines_mask]

            label=label[valid_lines_mask]

            #print("label after filtering", label.shape)

            u2v = xyu - xyv
            u2v /= torch.sqrt((u2v ** 2).sum(-1, keepdim=True)).clamp(min=1e-6)
            # feat = torch.cat(
            #     [
            #         xyu / 256 * M.use_cood,
            #         xyv / 256 * M.use_cood,
            #         u2v * M.use_slop,
            #         (u[:, None] > K).float(),
            #         (v[:, None] > K).float(),
            #     ],
            #     1,
            # )
            line = torch.cat([xyu[:, None], xyv[:, None]], 1)
            # print("lines sample:", line.shape)
            # print("label", label.shape)
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