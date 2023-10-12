import glob
import json
import math
import os
import random

import numpy as np
import numpy.linalg as LA
import torch
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from lcnn.config import M


class WireframeDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        filelist = glob.glob(f"{rootdir}/{split}/*_label.npz")
        filelist.sort()

        print(f"n{split}:", len(filelist))
        self.split = split
        self.filelist = filelist

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        iname = self.filelist[idx][:-10].replace("_a0", "").replace("_a1", "") + ".png"
        image = io.imread(iname, as_gray=True).astype(float)
        image = image[:, :, np.newaxis]
        if "a1" in self.filelist[idx]:
            image = image[:, ::-1, :]
        image = (image - M.image.mean) / M.image.stddev
        image = np.rollaxis(image, 2).copy()

        # npz["jmap"]: [J, H, W]    Junction heat map
        # npz["joff"]: [J, 2, H, W] Junction offset within each pixel
        # npz["lmap"]: [L, H, W]    Line heat map with anti-aliasing
        # npz["junc"]: [Na, 3]      Junction coordinates
        # npz["Lpos"]: [L, M, 2]       Positive lines represented with junction indices
        # npz["Lneg"]: [L, M, 2]       Negative lines represented with junction indices
        # npz["lpos"]: [L, Np, 2, 3]   Positive lines represented with junction coordinates
        # npz["lneg"]: [L, Nn, 2, 3]   Negative lines represented with junction coordinates
        #
        # For junc, lpos, and lneg that stores the junction coordinates, the last
        # dimension is (y, x, t), where t represents the type of that junction.
        with np.load(self.filelist[idx]) as npz:
            target = {
                name: torch.from_numpy(npz[name]).float()
                for name in ["jmap", "joff", "lmap"]
            }
            lpos = npz["lpos"].copy()

            lpos0 = slice_permute(lpos[0], M.n_stc_posl)
            lpos1 = slice_permute(lpos[1], M.n_stc_posl)

            lneg = npz["lneg"].copy()
            lneg0 = slice_permute(lneg[0], M.n_stc_negl)
            lneg1 = slice_permute(lneg[1], M.n_stc_negl)

            lpre = np.concatenate([lpos0, lpos1, lneg0, lneg1], 0)
            npos0, nneg0, npos1, nneg1 = len(lpos0), len(lneg0), len(lpos1), len(lneg1)


            labels_1 = torch.tensor([0, 1, 0]).float().repeat((npos0, 1))  # Class 1 for lpos0 dashed
            labels_2 = torch.tensor([0, 0, 1]).float().repeat((npos1, 1))  # Class 2 for lpos1 continous
            labels_0 = torch.tensor([1, 0, 0]).float().repeat((nneg0 + nneg1, 1))  # Class 0 for nneg all
            lpre_label = torch.cat([labels_1, labels_2, labels_0], dim=0)


            for i in range(len(lpre)):
                if random.random() > 0.5:
                    lpre[i] = lpre[i, ::-1]
            # ldir = lpre[:, 0, :2] - lpre[:, 1, :2]
            # ldir /= np.clip(LA.norm(ldir, axis=1, keepdims=True), 1e-6, None)
            # feat = [
            #     lpre[:, :, :2].reshape(-1, 4) / 256 * M.use_cood,
            #     ldir * M.use_slop,
            #     lpre[:, :, 2],
            # ]
            feat = np.concatenate(feat, 1)
            meta = {
                "junc": torch.from_numpy(npz["junc"][:, :2]),
                "jtyp": torch.from_numpy(npz["junc"][:, 2]).byte(),
                "Lpos": self.adjacency_matrix(len(npz["junc"]), npz["Lpos"][0], npz["Lpos"][1]),
                "Lneg": self.adjacency_matrix(len(npz["junc"]), npz["Lneg"][0], npz["Lneg"][1]),
                "lpre": torch.from_numpy(lpre[:, :, :2]),
                "lpre_label": lpre_label,
                #"lpre_feat": torch.from_numpy(feat),
            }
            # for key, value in meta.items():
            #     print(f"{key}: {value.shape}")
        return torch.from_numpy(image).float(), meta, target

    def adjacency_matrix(self, n, link1, link2):
        mat = torch.zeros(n + 1, n + 1, dtype=torch.uint8)
        link1 = torch.from_numpy(link1)
        link2 = torch.from_numpy(link2)

        if len(link1) > 0:
            mat[link1[:, 0], link1[:, 1]] = 1
            mat[link1[:, 1], link1[:, 0]] = 1

        if len(link2) > 0:
            mat[link2[:, 0], link2[:, 1]] = 2
            mat[link2[:, 1], link2[:, 0]] = 2

        return mat


import numpy as np


def slice_permute(lpos, n_stc_posl):
    # Ensure lpos is a NumPy array
    lpos = np.array(lpos)
    # Check and remove slices containing -1
    contains_neg_one = np.any(lpos == -1, axis=(1, 2))
    lpos_processed = lpos[~contains_neg_one]
    # Check if lpos_processed is empty
    if lpos_processed.size == 0:
        print("Warning: All slices removed, returning empty array.")
    return lpos_processed

    # Permute the slices
    lpos_processed = np.random.permutation(lpos_processed)

    # Ensure n_stc_posl is a valid index
    if n_stc_posl < lpos_processed.shape[0]:
        lpos_processed = lpos_processed[:n_stc_posl]

    return lpos_processed


def collate(batch):
    return (
        default_collate([b[0] for b in batch]),
        [b[1] for b in batch],
        default_collate([b[2] for b in batch]),
    )