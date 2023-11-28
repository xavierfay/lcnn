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

def resize_image_binary(img, new_size, upscale_factor=4):


    upscale_width = img.shape[1] * upscale_factor
    upscale_height = img.shape[0] * upscale_factor
    img = cv2.resize(img, (upscale_width, upscale_height), interpolation=cv2.INTER_NEAREST)

    # Downscale to the desired size using bilinear interpolation
    while img.shape[1] > new_size[0] * 2 and img.shape[0] > new_size[1] * 2:
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_CUBIC)

    img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
    # Binarize the image
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Save the resized image
    return binary_img


def adjacency_matrix(n, *links):
    mat = np.zeros((n + 1, n + 1))

    for idx, link in enumerate(links, 1):  # idx will start from 1, corresponding to link1, link2, etc.
        if len(link) > 0:
            mat[link[:, 0], link[:, 1]] = idx
            mat[link[:, 1], link[:, 0]] = idx

    return mat
def save_heatmap(prefix, image, lines, classes):
    im_rescale = (1536, 1536)
    heatmap_scale = (128, 128)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]
    jmap = np.zeros((34,) + heatmap_scale, dtype=np.float32)
    joff = np.zeros((34, 2) + heatmap_scale, dtype=np.float32)
    lmap = np.zeros((3,)+ heatmap_scale, dtype=np.float32)


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

    lnid = [[], [], [], []]
    l_label = []
    lpos, lneg = [], []

    for i, line in enumerate(lines):
        v0, v1 = line[0], line[1]
        lclass = classes[i]-1
        lnid[lclass].append((jid(v0), jid(v1)))
        lpos.append([junc[jid(v0)], junc[jid(v1)]])
        l_label.append(classes[i])


        vint0, vint1 = to_int(v0[:2]), to_int(v1[:2])
        jmap[int(v0[2] - 1)][vint0] = 1
        jmap[int(v1[2] - 1)][vint1] = 1
        rr, cc, value = skimage.draw.line_aa(*to_int(v0[:2]), *to_int(v1[:2]))
        lmap[lclass, rr, cc] = np.maximum(lmap[lclass, rr, cc], value)

    for v in junc:
        vint = to_int(v[:2])
        offset = v[:2] - vint - 0.5
        for layer in range(34):  # Assuming you want to update the first three layers
            joff[layer, :, vint[0], vint[1]] = offset

    lineset = set([frozenset(l) for l in lnid])

    for i0, i1 in combinations(range(len(junc)), 2):
        if frozenset([i0, i1]) not in lineset:
            v0, v1 = junc[i0], junc[i1]
            rr, cc, value = skimage.draw.line_aa(*to_int(v0[:2]), *to_int(v1[:2]))
            avg_value = np.average(np.minimum(value, lmap[0, rr, cc]))
            lneg.append([v0, v1, i0, i1, avg_value])

    assert len(lneg) != 0
    lneg.sort(key=lambda l: -l[-1])

    junc = np.array(junc, dtype=np.float32)
    Lpos = adjacency_matrix(len(junc), np.array(lnid[0]), np.array(lnid[1]), np.array(lnid[2]), np.array(lnid[3]))
    Lneg = np.array([l[2:4] for l in lneg][:4000], dtype=int)
    Lneg = adjacency_matrix(len(junc), Lneg)
    lpos = np.array(lpos, dtype=np.float32)
    l_label = np.array(l_label, dtype=int)
    lneg = np.array([l[:2] for l in lneg[:2000]], dtype=np.float32)

    image = resize_image_binary(image, im_rescale)

    np.savez_compressed(
        f"{prefix}_label.npz",
        aspect_ratio=image.shape[1] / image.shape[0],
        jmap=jmap,  # [J, H, W]       Junction heat map
        joff=joff,  # [J, 2, H, W]    Junction offset within each pixel
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

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    left_top_corner = (0, 0)
    right_bottom_corner = (5665, 4325)

    # Crop the image
    img = img[left_top_corner[1]:right_bottom_corner[1], left_top_corner[0]:right_bottom_corner[0]]

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
    for batch in [ "test_complete"]:
        anno_file = os.path.join(data_root, f"{batch}.json")

        with open(anno_file, "r") as f:
            dataset = json.load(f)

        args_to_pass = [(data_item, data_root, data_output, batch) for data_item in dataset]
        results = parmap(handle_wrapper, args_to_pass, 16)




if __name__ == "__main__":
    main()

