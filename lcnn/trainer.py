import atexit
import os
import os.path as osp
import shutil
import signal
import subprocess
import threading
import time
from timeit import default_timer as timer

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from tensorboardX import SummaryWriter

from lcnn.config import C, M
from lcnn.utils import recursive_to

import wandb
from torch.cuda.amp import autocast, GradScaler

class Trainer(object):
    def __init__(self, device, model, optimizer, train_loader, val_loader, out):
        self.device = device
        if M.use_half and device == torch.device("cuda"):
            model = model.half()
        self.model = model
        self.optim = optimizer
        self.scaler = GradScaler()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = C.model.batch_size

        self.validation_interval = C.io.validation_interval

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        ### unable to run tensorboard on the cluster
        # self.run_tensorboard()
        # time.sleep(1)
        self.board_out = osp.join(self.out, "tensorboard")
        if not osp.exists(self.board_out):
            os.makedirs(self.board_out)
        self.writer = SummaryWriter(self.board_out)


        self.epoch = 0
        self.iteration = 0
        self.max_epoch = C.optim.max_epoch
        self.lr_decay_epoch = C.optim.lr_decay_epoch
        self.num_stacks = C.model.num_stacks
        self.mean_loss = self.best_mean_loss = 1e1000

        self.loss_labels = None
        self.avg_metrics = None
        self.metrics = np.zeros(0)

        if M.use_wandb:
            mode = 'online'
        else:
            mode = 'disabled'

        wandb.init(project='line_reader', entity='xfung', mode=mode)
        config = wandb.config
        config.update(M, allow_val_change=True)

    # def run_tensorboard(self):
    #     board_out = osp.join(self.out, "tensorboard")
    #     if not osp.exists(board_out):
    #         os.makedirs(board_out)
    #     self.writer = SummaryWriter(board_out)
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ""
    #     p = subprocess.Popen(
    #         ["tensorboard", f"--logdir={board_out}", f"--port={C.io.tensorboard_port}"]
    #     )
    #
    #     def killme():
    #         os.kill(p.pid, signal.SIGTERM)
    #
    #     atexit.register(killme)

    def _loss(self, result):
        losses = result["losses"]
        # Don't move loss label to other place.
        # If I want to change the loss, I just need to change this function.
        if self.loss_labels is None:
            self.loss_labels = ["sum"] + list(losses[0].keys())
            self.metrics = np.zeros([self.num_stacks, len(self.loss_labels)])
            print()
            print(
                "| ".join(
                    ["progress "]
                    + list(map("{:7}".format, self.loss_labels))
                    + ["speed"]
                )
            )
            with open(f"{self.out}/loss.csv", "a") as fout:
                print(",".join(["progress"] + self.loss_labels), file=fout)

        total_loss = 0
        for i in range(self.num_stacks):
            for j, name in enumerate(self.loss_labels):
                if name == "sum":
                    continue
                if name not in losses[i]:
                    assert i != 0
                    continue
                loss = losses[i][name].mean()
                self.metrics[i, 0] += loss.item()
                self.metrics[i, j] += loss.item()
                total_loss += loss
        return total_loss

    def validate(self):
        tprint("Running validation...", " " * 75)
        training = self.model.training
        self.model.eval()

        viz = osp.join(self.out, "viz", f"{self.iteration * M.batch_size_eval:09d}")
        npz = osp.join(self.out, "npz", f"{self.iteration * M.batch_size_eval:09d}")
        osp.exists(viz) or os.makedirs(viz)
        osp.exists(npz) or os.makedirs(npz)

        total_loss = 0
        self.metrics[...] = 0
        with torch.no_grad():
            for batch_idx, (image, meta, target) in enumerate(self.val_loader):
                input_dict = {
                    "image": recursive_to(image, self.device),
                    "meta": recursive_to(meta, self.device),
                    "target": recursive_to(target, self.device),
                    "mode": "validation",
                }

                if M.use_half and self.device == torch.device("cuda"):
                    input_dict["image"] = input_dict["image"].half()
                    if isinstance(input_dict["meta"], list):
                        input_dict["meta"] = [item.half() if torch.is_tensor(item) else item for item in
                                              input_dict["meta"]]
                    if torch.is_tensor(input_dict["target"]):
                        input_dict["target"] = input_dict["target"].half()

                # Using autocast for the forward pass
                with autocast():
                    result = self.model(input_dict)
                    H = result["preds"]

                total_loss += self._loss(result)

                for i in range(H["jmap"].shape[0]):
                    index = batch_idx * M.batch_size_eval + i
                    np.savez(
                        f"{npz}/{index:06}.npz",
                        **{k: v[i].cpu().numpy() for k, v in H.items()},
                    )

                    if index >= 20:
                        continue
                    self._plot_samples(i, index, H, meta, target, f"{viz}/{index:06}")

        self._write_metrics(len(self.val_loader), total_loss, "validation", True)
        self.mean_loss = total_loss / len(self.val_loader)

        checkpoint_path = osp.join(self.out, "checkpoint_latest.pth")
        torch.save({
            "iteration": self.iteration,
            "arch": self.model.__class__.__name__,
            "optim_state_dict": self.optim.state_dict(),
            "model_state_dict": self.model.state_dict(),
            "best_mean_loss": self.best_mean_loss,
        }, checkpoint_path)
        wandb.save(checkpoint_path)
        shutil.copy(
            osp.join(self.out, "checkpoint_latest.pth"),
            osp.join(npz, "checkpoint.pth"),
        )
        if self.mean_loss < self.best_mean_loss:
            self.best_mean_loss = self.mean_loss
            shutil.copy(
                osp.join(self.out, "checkpoint_latest.pth"),
                osp.join(self.out, "checkpoint_best.pth"),
            )

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        time = timer()
        for batch_idx, (image, meta, target) in enumerate(self.train_loader):

            self.optim.zero_grad()
            self.metrics[...] = 0

            input_dict = {
                "image": recursive_to(image, self.device),
                "meta": recursive_to(meta, self.device),
                "target": recursive_to(target, self.device),
                "mode": "training",
            }
            if M.use_half and self.device == torch.device("cuda"):
                input_dict["image"] = input_dict["image"].half()
                if isinstance(input_dict["meta"], list):
                    input_dict["meta"] = [item.half() if torch.is_tensor(item) else item for item in input_dict["meta"]]
                if torch.is_tensor(input_dict["target"]):
                    input_dict["target"] = input_dict["target"].half()

            with autocast():
                result = self.model(input_dict)
                loss = self._loss(result)

            if np.isnan(loss.item()):
                print("loss is nan while training")

            wandb.log({"grad_norm": torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)},
                      step=self.iteration)

            # Backward pass and optimizer step with scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad()

            if self.avg_metrics is None:
                self.avg_metrics = self.metrics
            else:
                self.avg_metrics = self.avg_metrics * 0.9 + self.metrics * 0.1
            self.iteration += 1
            self._write_metrics(1, loss.item(), "training", do_print=False)

            if self.iteration % 4 == 0:
                pprint(
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, self.avg_metrics[0]))
                    + f"| {4 * self.batch_size / (timer() - time):04.1f} "
                )
                time = timer()
            # num_images = self.batch_size * self.iteration
            # if num_images % self.validation_interval == 0 or num_images == 600:
            #     self.validate()
            #     time = timer()

    def _write_metrics(self, size, total_loss, prefix, do_print=False):
        for i, metrics in enumerate(self.metrics):
            for label, metric in zip(self.loss_labels, metrics):
                wandb.log(
                    {f"{prefix}/{i}/{label}": metric / size for i, metrics in enumerate(self.metrics) for label, metric
                     in zip(self.loss_labels, metrics)}, step=self.iteration)
                wandb.log({f"{prefix}/total_loss": total_loss / size}, step=self.iteration)
                self.writer.add_scalar(
                    f"{prefix}/{i}/{label}", metric / size, self.iteration
                )
            if i == 0 and do_print:
                csv_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size:07},"
                    + ",".join(map("{:.11f}".format, metrics / size))
                )
                prt_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, metrics / size))
                )
                with open(f"{self.out}/loss.csv", "a") as fout:
                    print(csv_str, file=fout)
                pprint(prt_str, " " * 7)
        self.writer.add_scalar(
            f"{prefix}/total_loss", total_loss / size, self.iteration
        )
        return total_loss

    def _plot_samples(self, i, index, result, meta, target, prefix):
        # for key, value in result.items():
        #     if isinstance(value, (torch.Tensor, np.ndarray)):
        #         print(f"plot sample function {key}: {value.shape}")
        #     else:
        #         print(f"{key} is a {type(value)}, so it doesn't have a shape attribute")

        fn = self.val_loader.dataset.filelist[index][:-10].replace("_a0", "") + ".png"
        img = io.imread(fn)
        imshow(img), plt.savefig(f"{prefix}_img.jpg"), plt.close()

        mask_result = result["jmap"][i].cpu().numpy()
        mask_result = np.sum(mask_result, axis=0)
        #mask_result = plt_heatmaps(mask_result)
        mask_target = target["jmap"][i].cpu().numpy()
        mask_target = plt_heatmaps(mask_target)


        # Displaying the results using the updated imshow function
        imshow(mask_result, cmap="jet"),  plt.savefig(f"{prefix}_mask_b.jpg"), plt.close()
        imshow(mask_target, cmap="jet"), plt.savefig(f"{prefix}_mask_a.jpg"), plt.close()

        # imshow(mask_target), plt.savefig(f"{prefix}_mask_a.jpg"), plt.close()
        # imshow(mask_result), plt.savefig(f"{prefix}_mask_b.jpg"), plt.close()

        for j, results in enumerate(result["lmap"][i]):
            line_result = results.cpu().numpy()
            imshow(line_result, cmap="hot"), plt.savefig(f"{prefix}_line_{j}b.jpg"), plt.close()

        for j, target in enumerate(target["lmap"][i]):
            line_target = target.cpu().numpy()
            imshow(line_target, cmap="hot"), plt.savefig(f"{prefix}_line_{j}a.jpg"), plt.close()

        def draw_vecl(lines, sline, juncs, jtyp, fn):
            imshow(img)

            if len(lines) > 0 and not (lines[0] == 0).all():
                printed_count = 0
                # print("This is the shape of lines", lines.shape)
                for i, ((a, b), s) in enumerate(zip(lines, sline)):

                    line_type = np.argmax(s)
                    if i > 0 and (lines[i] == lines[0]).all() and (sline[i] == sline[0]).all():
                        print("broken because double line")
                        print("printed count", printed_count)
                        break
                    printed_count += 1
                    if line_type == 0:
                        #plt.plot([a[1], b[1]], [a[0], b[0]], c="green", linewidth=4)
                        continue
                    elif line_type == 1:
                        plt.plot([a[1], b[1]], [a[0], b[0]], c=c(np.max(s)), linewidth=4, linestyle='--')
                    elif line_type == 2:
                        plt.plot([a[1], b[1]], [a[0], b[0]], c=c(np.max(s)), linewidth=4)
                    elif line_type == 3:
                        plt.plot([a[1], b[1]], [a[0], b[0]], c="blue", linewidth=4)
                    else:
                        plt.plot([a[1], b[1]], [a[0], b[0]], c="red", linewidth=4)
                        print(line_type)

            if not (juncs[0] == 0).all():
                for i, j in enumerate(juncs):
                    if i > 0 and (i == juncs[0]).all():
                        break
                    if jtyp[i] == 1:
                        plt.scatter(j[1], j[0], c="red", s=64, zorder=100)
                    # elif jtyp[i] == 2:
                    #     plt.scatter(j[1], j[0], c="yellow", s=64, zorder=100)
                    # else:
                    #     # add plot with number from jtype
                    #     plt.scatter(j[1], j[0], c="blue", s=64, zorder=100)
                    #     plt.text(j[1] + 10, j[0], str(jtyp[i]), color="black", fontsize=12, zorder=200)

            # Display a dummy colorbar for the line colors
            dummy_image = np.array([[0, 1]])  # 2D image with values 0 and 1
            im_dummy = plt.imshow(dummy_image, cmap='jet', visible=False)  # use the same colormap as lines
            plt.colorbar(im_dummy, orientation='vertical', fraction=0.046)

            plt.savefig(fn), plt.close()

        juncs = meta[i]["junc"].cpu().numpy() * 4
        jtyp = meta[i]["jtyp"].cpu().numpy()


        rjuncs = result["juncs"][i].cpu().numpy() * 4
        rjtyp = result["jtype"][i].cpu().numpy()


        lpre = meta[i]["lpre"].cpu().numpy() * 4
        lpre_label = meta[i]["lpre_label"].cpu().numpy()
        #print("vecl target max", np.max(lpre_label), lpre_label)
        vecl_result = result["lines"][i].cpu().numpy() * 4
        print("lines in trainer:",vecl_result.shape, vecl_result[1])
        #print("results for lines",vecl_result.shape, vecl_result[1])
        score = result["score"][i].cpu().numpy()
        #print("score =", np.max(score), score)

        # for i in range(1,2):
        #     lpre = lpre[vecl_target == i]
        #     draw_vecl(lpre, np.ones(lpre.shape[0]), juncs, junts, f"{prefix}_vecl_{i}a.jpg")
        draw_vecl(lpre, lpre_label, juncs, jtyp, f"{prefix}_vecl_a.jpg")
        draw_vecl(vecl_result, score, rjuncs, rjtyp, f"{prefix}_vecl_b.jpg")

    def train(self):
        plt.rcParams["figure.figsize"] = (24, 24)
        # if self.iteration == 0:
        #     self.validate()
        epoch_size = len(self.train_loader)
        start_epoch = self.iteration // epoch_size
        for self.epoch in range(start_epoch, self.max_epoch):
            if self.epoch == self.lr_decay_epoch:
                self.optim.param_groups[0]["lr"] /= 10
            self.train_epoch()
            self.validate()



cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


def imshow(im, cmap="gray"):
    plt.close()
    plt.tight_layout()
    plt.imshow(im, cmap=cmap)
    plt.colorbar(fraction=0.046)
    plt.xlim([0, im.shape[0]])
    plt.ylim([im.shape[0], 0])


def plt_heatmaps(jmap):
    # Define a colormap of 34 unique colors (you can customize this)
    colormap = plt.cm.jet(np.linspace(0, 1, 34))

    # Create an image of shape 256x256x3 initialized with ones to have a white background
    combined_image = np.ones((256, 256, 3))

    for i in range(jmap.shape[0]):
        # Multiply each heatmap layer with its corresponding color
        colored_layer = np.expand_dims(jmap[i], axis=-1) * (colormap[i, :3] - 1)
        combined_image += colored_layer

    # Clip values between 0 and 1 for visualization
    combined_image = np.clip(combined_image, 0, 1)

    return combined_image

def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)


# def _launch_tensorboard(board_out, port, out):
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""
#     p = subprocess.Popen(["tensorboard", f"--logdir={board_out}", f"--port={port}"])
#
#     def kill():
#         os.kill(p.pid, signal.SIGTERM)
#
#     atexit.register(kill)