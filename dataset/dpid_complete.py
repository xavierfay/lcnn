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

import matplotlib.pyplot as plt

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


def pad_and_convert_to_array(lpos):
    max_len = max(len(sublist) for sublist in lpos)

    # Determine the shape of individual data points
    if lpos:
        if lpos[0]:  # If lpos[0] is not empty/0, use it to determine data_shape
            data_shape = np.asarray(lpos[0][0]).shape
        elif len(lpos) > 1 and lpos[1]:  # If lpos[0] is empty/0 and lpos[1] exists and is not empty, use lpos[1]
            data_shape = np.asarray(lpos[1][0]).shape
        else:
            print(lpos)
            raise ValueError("Unable to determine data_shape from lpos[0] and lpos[1].")
    else:
        print(lpos)
        raise ValueError("lpos is empty, cannot determine data_shape.")

    # Initialize an array filled with -1 with appropriate shape
    padded_lpos = -1 * np.ones((len(lpos), max_len) + data_shape, dtype=np.float32)

    # Populate the array with available data
    for i, sublist in enumerate(lpos):
        for j, data_point in enumerate(sublist):
            padded_lpos[i, j] = data_point

    return padded_lpos

def save_heatmap(prefix, image, lines, classes):
    im_rescale = (1024, 1024)
    heatmap_scale = (256, 256)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]
    jmap = np.zeros((1,) + heatmap_scale, dtype=np.float32)
    joff = np.zeros((2, 2) + heatmap_scale, dtype=np.float32)
    lmap = np.zeros(heatmap_scale, dtype=np.float32)

    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    lines[:, :, :2] = lines[:, :, 1::-1]

    junc = []
    jids = {}

    def jid(jun):
        jun = tuple(jun[:3])
        if jun in jids:
            return jids[jun]
        jids[jun] = len(junc)
        junc.append(np.array(jun))
        return len(junc) - 1

    lnid, l_label = [], []
    lpos, lneg = [], []
    num_classes = max(classes) + 1
    #print("Number of classes", num_classes)
    #num_classes = 2
    # # Initialize sublists
    # lnid = [[] for _ in range(num_classes)]
    # lpos = [[] for _ in range(num_classes)]
    # lneg = []

    for i, line in enumerate(lines):
        v0, v1 = line[0], line[1]
        lclass = classes[i]
        lnid.append((jid(v0), jid(v1)))
        lpos.append([junc[jid(v0)], junc[jid(v1)]])
        l_label.append(lclass)

        vint0, vint1 = to_int(v0[:2]), to_int(v1[:2])
        jmap[0][vint0] = int(v0[2])  # assuming v0[2] gives the correct index in the first dimension and the value to set is 1
        jmap[0][vint1] = int(v1[2])  # assuming v1[2] gives the correct index in the first dimension and the value to set is 1
        #print("v0", v0, "v1", v1, "vint0", vint0, "vint1", vint1, "lclass", lclass)
        rr, cc, value = skimage.draw.line_aa(*to_int(v0[:2]), *to_int(v1[:2]))

        lmap[rr, cc] = np.maximum(lmap[rr, cc], value*lclass)

    for v in junc:
        vint = to_int(v[:2])
        joff[0, :, vint[0], vint[1]] = v[:2] - vint - 0.5
    llmap = zoom(lmap, [0.5, 0.5])
    lineset = set([frozenset(l) for l in lnid])

    for i0, i1 in combinations(range(len(junc)), 2):
        if frozenset([i0, i1]) not in lineset:
            v0, v1 = junc[i0], junc[i1]
            vint0, vint1 = to_int(v0[:2] / 2), to_int(v1[:2] / 2)
            rr, cc, value = skimage.draw.line_aa(*vint0, *vint1)
            avg_value = np.average(np.minimum(value, llmap[rr, cc]))
            lneg.append([v0, v1, i0, i1, avg_value])

    assert len(lneg) != 0
    lneg.sort(key=lambda l: -l[-1])

    junc = np.array(junc, dtype=np.float32)
    Lpos = np.array(lnid, dtype=int)
    Lneg = np.array([l[2:4] for l in lneg][:4000], dtype=int)
    lpos = np.array(lpos, dtype=np.float32)
    l_label = np.array(l_label, dtype=int)
    lneg = np.array([l[:2] for l in lneg[:2000]], dtype=np.float32)

    image = cv2.resize(image, im_rescale)

    np.savez_compressed(
        f"{prefix}_label.npz",
        aspect_ratio=image.shape[1] / image.shape[0],
        jmap=jmap,  # [J, H, W]       Junction heat map
        joff=joff,  # [J, 2, H, W]    Junction offset within each pixel
        lmap=lmap,  # [L, H, W]       Line heat map with anti-aliasing
        junc=junc,  # [Na, 3]         Junction coordinate
        Lpos=Lpos,  # [L, M, 2]       Positive lines represented with junction indices
        Lneg=Lneg,  # [L, M, 2]       Negative lines represented with junction indices
        lpos=lpos,  # [L, Np, 2, 3]   Positive lines represented with junction coordinates
        l_label=l_label,  # [L, Nn, 2, 3]   Negative lines represented with junction coordinates
        lneg=lneg,  # [L, Nn, 2, 3]   Negative lines represented with junction coordinates
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

    # Separate coordinates and classes
    lines_raw = np.array(data["lines"])
    # Separate flags
    flags = lines_raw[:, 0]
    # Get coordinates and reshape them
    coordinates = lines_raw[:, 1:].reshape(-1, 2, 3)

    # Combine flags and reshaped coordinates into a structured array if you want
    lines = np.zeros(coordinates.shape[0], dtype=[('flag', int), ('coordinates', int, (2, 3))])
    lines['flag'] = flags
    lines['coordinates'] = coordinates

    os.makedirs(os.path.join(data_output, batch), exist_ok=True)

    # Your transformations
    lines0 = coordinates.copy()
    lines1 = coordinates.copy()
    lines1[:, :, 0] = img.shape[1] - lines1[:, :, 0]
    lines2 = coordinates.copy()
    lines2[:, :, 1] = img.shape[0] - lines2[:, :, 1]
    lines3 = coordinates.copy()
    lines3[:, :, 0] = img.shape[1] - lines3[:, :, 0]
    lines3[:, :, 1] = img.shape[0] - lines3[:, :, 1]

    path = os.path.join(data_output, batch, prefix)
    classes = lines["flag"]+1
    # Pass classes to save_heatmap
    save_heatmap(f"{path}_0", img[::, ::], lines0, classes)
    if batch == "train_complete":
        save_heatmap(f"{path}_1", img[::, ::-1], lines1, classes)
        save_heatmap(f"{path}_2", img[::-1, ::], lines2, classes)
        save_heatmap(f"{path}_3", img[::-1, ::-1], lines3, classes)

    print("Finishing", os.path.join(data_output, batch, prefix))


def main():
    args = docopt(__doc__)
    data_root = args["<src>"]
    data_output = args["<dst>"]

    os.makedirs(data_output, exist_ok=True)
    for batch in [ "train_complete", "test_complete"]:
        anno_file = os.path.join(data_root, f"{batch}.json")

        with open(anno_file, "r") as f:
            dataset = json.load(f)

        args_to_pass = [(data_item, data_root, data_output, batch) for data_item in dataset]
        results = parmap(handle_wrapper, args_to_pass, 16)

if __name__ == "__main__":
    main()

