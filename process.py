#!/usr/bin/env python3
"""Process a dataset with the trained neural network
Usage:
    process.py [options] <yaml-config> <checkpoint> <image-dir> <output-dir>
    process.py (-h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint
   <image-dir>                   Path to the directory containing processed images
   <output-dir>                  Path to the output directory

Options:
   -h --help                     Show this screen.
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
   --plot                        Plot the result
"""

import os
import sys
import shlex
import pprint
import random
import os.path as osp
import threading
import subprocess

import yaml
import numpy as np
import torch
import matplotlib as mpl
import skimage.io
import matplotlib.pyplot as plt
from docopt import docopt

import lcnn
from lcnn.utils import recursive_to
from lcnn.config import C, M
from lcnn.datasets import WireframeDataset, collate
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
import scipy.io as sio
from lcnn.models.HT import hough_transform

def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/wireframe.yaml"
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    print("check")
    if os.path.isfile(C.io.vote_index):
        print('load vote_index ... ')
        # Check if the file is .mat or .npy format
        if C.io.vote_index.endswith('.mat'):
            vote_index = sio.loadmat(C.io.vote_index)['vote_index']
        elif C.io.vote_index.endswith('.npy'):
            vote_index = np.load(C.io.vote_index)
        else:
            raise ValueError("Unknown file format for vote_index.")
    else:
        print('compute vote_index ... ')
        vote_index = hough_transform(rows=256, cols=256, theta_res=3, rho_res=1)
        # Save as .npy format
        np.save(C.io.vote_index.replace('.mat', '.npy'), vote_index)
    print(type(vote_index))
    vote_index = torch.from_numpy(vote_index).float().contiguous().to(device)
    if M.backbone == "stacked_hourglass":
        model = lcnn.models.hg(
            depth=M.depth,
            head=MultitaskHead,
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),
            vote_index=vote_index,
        )
    else:
        raise NotImplementedError

    checkpoint = torch.load(args["<checkpoint>"], map_location=torch.device("cpu"))
    model = MultitaskLearner(model)
    model = LineVectorizer(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    loader = torch.utils.data.DataLoader(
        WireframeDataset(args["<image-dir>"], split="test_complete"),
        shuffle=False,
        batch_size=M.batch_size,
        collate_fn=collate,
        num_workers=C.io.num_workers if os.name != "nt" else 0,
        pin_memory=True,
    )
    os.makedirs(args["<output-dir>"], exist_ok=True)

    for batch_idx, (image, meta, target) in enumerate(loader):
        with torch.no_grad():
            input_dict = {
                "image": recursive_to(image, device),
                "meta": recursive_to(meta, device),
                "target": recursive_to(target, device),
                "mode": "validation",
            }
            H = model(input_dict)["preds"]
            for i in range(M.batch_size):
                index = batch_idx * M.batch_size + i
                np.savez(
                    osp.join(args["<output-dir>"], f"{index:06}.npz"),
                    **{k: v[i].cpu().numpy() for k, v in H.items()},
                )
                if not args["--plot"]:
                    continue
                im = image[i].cpu().numpy().transpose(1, 2, 0)
                im = im * M.image.stddev + M.image.mean
                lines = H["lines"][i].cpu().numpy() * 4
                scores = H["score"][i].cpu().numpy()
                if len(lines) > 0 and not (lines[0] == 0).all():
                    for i, ((a, b), s) in enumerate(zip(lines, scores)):
                        if i > 0 and (lines[i] == lines[0]).all():
                            break
                        plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=4)
                plt.show()


cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


if __name__ == "__main__":
    main()