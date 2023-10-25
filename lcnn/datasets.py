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
                for name in ["lmap", "joff", "jmap"]
            }


            lpos_indices = np.random.permutation(len(npz["lpos"]))[: M.n_stc_posl0 + M.n_stc_posl1 + M.n_stc_posl2]
            lneg_indices = np.random.permutation(len(npz["lneg"]))[: M.n_stc_negl]

            # Use these indices to get lpos and lneg
            lpos = npz["lpos"][lpos_indices]
            lneg = npz["lneg"][lneg_indices]

            # Get the labels corresponding to the selected indices
            lpos_label = npz["l_label"].copy()
            lpos_label = lpos_label[lpos_indices]
            lneg_label = np.zeros(len(lneg_indices))

            # Concatenate to get lpre_label of the same length as lpre
            lpre_label = np.concatenate([lpos_label, lneg_label], 0)

            num_classes = int(np.max(lpre_label)) + 1
            lpre_label = np.eye(num_classes)[lpre_label.astype(int)]

            # Concatenate to get lpre
            lpre = np.concatenate([lpos, lneg], 0)

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
            # feat = np.concatenate(feat, 1)
            meta = {
                "junc": torch.from_numpy(npz["junc"][:, :2]),
                "jtyp": torch.from_numpy(npz["junc"][:, 2]).byte(),
                "Lpos": torch.from_numpy(npz["Lpos"]),
                "Lneg": torch.from_numpy(npz["Lneg"]),
                "lpre": torch.from_numpy(lpre[:, :, :2]),
                "lpre_label": torch.from_numpy(lpre_label),
                # "lpre_feat": torch.from_numpy(feat),
            }
            # for key, value in meta.items():
            #     print(f"{key}: {value.shape}")
            # for key, value in target.items():
            #     print(f"{key}: {value.shape}")
        image_tensor = torch.from_numpy(image).float()
        if M.use_half:
            image_tensor = image_tensor.half()
            target = {key: value.half() for key, value in target.items()}
            meta = {key: value.half() if torch.is_tensor(value) else value for key, value in meta.items()}

        return image_tensor, meta, target

    def adjacency_matrix(self, n, link):
        mat = torch.zeros(n + 1, n + 1, dtype=torch.uint8)
        link = torch.from_numpy(link)
        if len(link) > 0:
            mat[link[:, 0], link[:, 1]] = 1
            mat[link[:, 1], link[:, 0]] = 1
        return mat




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