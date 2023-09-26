#!/usr/bin/env python
"""Process Huang's wireframe dataset for L-CNN network
Usage:
    dataset/wireframe.py <src> <dst>
    dataset/wireframe.py (-h | --help )

Examples:
    python dataset/wireframe.py /datadir/wireframe data/wireframe

Arguments:
    <src>                Original data directory of Huang's wireframe dataset
    <dst>                Directory of the output

Options:
   -h --help             Show this screen.
"""

import os
import sys
import json
from itertools import combinations

import cv2
import numpy as np
import skimage.draw
from docopt import docopt
from scipy.ndimage import zoom

try:
    sys.path.append(".")
    sys.path.append("..")
    from lcnn.utils import parmap
except Exception:
    raise


def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))


def save_heatmap(prefix, image, lines):
    im_rescale = (1024, 1024)
    heatmap_scale = (256, 256)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]
    jmap = np.zeros((3,) + heatmap_scale, dtype=np.float32)
    joff = np.zeros((3, 2) + heatmap_scale, dtype=np.float32)
    lmap = np.zeros(heatmap_scale, dtype=np.float32)


    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)

    junc = []
    jids = {}

    def jid(jun):
        jun = tuple(jun[:3])
        if jun in jids:
            return jids[jun]
        jids[jun] = len(junc)
        junc.append(np.array(jun))
        return len(junc) - 1

    lnid = []
    lpos, lneg = [], []
    for v0, v1 in lines:
        lnid.append((jid(v0), jid(v1)))
        lpos.append([junc[jid(v0)], junc[jid(v1)]])

        vint0, vint1 = to_int(v0[:2]), to_int(v1[:2])
        jmap[int(v0[2])][vint0] = 1  # assuming v0[2] gives the correct index in the first dimension and the value to set is 1
        jmap[int(v1[2])][vint1] = 1  # assuming v1[2] gives the correct index in the first dimension and the value to set is 1
        rr, cc, value = skimage.draw.line_aa(*to_int(v0[:2]), *to_int(v1[:2]))
        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

    for v in junc:
        vint = to_int(v[:2])
        joff[0, :, vint[0], vint[1]] = v[:2] - vint - 0.5
        joff[1, :, vint[0], vint[1]] = v[:2] - vint - 0.5

    llmap = zoom(lmap, [0.5, 0.5])
    lineset = set([frozenset(l) for l in lnid])
    for i0, i1 in combinations(range(len(junc)), 2):
        if frozenset([i0, i1]) not in lineset:
            v0, v1 = junc[i0], junc[i1]
            vint0, vint1 = to_int(v0[:2] / 2), to_int(v1[:2] / 2)
            rr, cc, value = skimage.draw.line_aa(*vint0, *vint1)
            lneg.append([v0, v1, i0, i1, np.average(np.minimum(value, llmap[rr, cc]))])

    assert len(lneg) != 0
    lneg.sort(key=lambda l: -l[-1])

    junc = np.array(junc, dtype=np.float32)
    Lpos = np.array(lnid, dtype=int)
    Lneg = np.array([l[2:4] for l in lneg][:4000], dtype=int)
    lpos = np.array(lpos, dtype=np.float32)
    lneg = np.array([l[:2] for l in lneg[:2000]], dtype=np.float32)

    image = cv2.resize(image, im_rescale)
    np.savez_compressed(
        f"{prefix}_label.npz",
        aspect_ratio=image.shape[1] / image.shape[0],
        jmap=jmap,  # [J, H, W]    Junction heat map
        joff=joff,  # [J, 2, H, W] Junction offset within each pixel
        lmap=lmap,  # [H, W]       Line heat map with anti-aliasing
        junc=junc,  # [Na, 3]      Junction coordinate
        Lpos=Lpos,  # [M, 2]       Positive lines represented with junction indices
        Lneg=Lneg,  # [M, 2]       Negative lines represented with junction indices
        lpos=lpos,  # [Np, 2, 3]   Positive lines represented with junction coordinates
        lneg=lneg,  # [Nn, 2, 3]   Negative lines represented with junction coordinates
    )
    cv2.imwrite(f"{prefix}.png", image)




def handle_wrapper(args):
    data, data_root, data_output, batch = args
    img = cv2.imread(os.path.join(data_root, "images", data["filename"]))
    if img is None:
        print(f"Failed to load image from {file_path}")
        return  # exit the function

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize the image using a threshold
    _, binarized = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Crop the image to 4/5 of its left size
    height, width = binarized.shape
    left = 0
    top = 0
    right = width * 4 // 5
    bottom = height
    img = binarized[top:bottom, left:right]



    prefix = data["filename"].split(".")[0]
    lines = np.array(data["lines"]).reshape(-1, 2, 3)
    os.makedirs(os.path.join(data_output, batch), exist_ok=True)

    lines0 = lines.copy()
    lines1 = lines.copy()
    lines1[:, :, 0] = img.shape[1] - lines1[:, :, 0]
    lines2 = lines.copy()
    lines2[:, :, 1] = img.shape[0] - lines2[:, :, 1]
    lines3 = lines.copy()
    lines3[:, :, 0] = img.shape[1] - lines3[:, :, 0]
    lines3[:, :, 1] = img.shape[0] - lines3[:, :, 1]

    path = os.path.join(data_output, batch, prefix)
    save_heatmap(f"{path}_0", img[::, ::], lines0)
    if batch != "train":
        save_heatmap(f"{path}_1", img[::, ::-1], lines1)
        save_heatmap(f"{path}_2", img[::-1, ::], lines2)
        save_heatmap(f"{path}_3", img[::-1, ::-1], lines3)
    print("Finishing", os.path.join(data_output, batch, prefix))


def main():
    args = docopt(__doc__)
    data_root = args["<src>"]
    data_output = args["<dst>"]

    os.makedirs(data_output, exist_ok=True)
    for batch in ["valid_twoclass", "train_twoclass"]:
        anno_file = os.path.join(data_root, f"{batch}.json")

        with open(anno_file, "r") as f:
            dataset = json.load(f)

        args_to_pass = [(data_item, data_root, data_output, batch) for data_item in dataset]
        results = parmap(handle_wrapper, args_to_pass, 16)

if __name__ == "__main__":
    main()

