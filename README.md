# PandID-Net "Robust line detection and association in P\&IDs"


This reposisory contains the inp
This repository contains the official PyTorch implementation of the paper:  *[Yichao Zhou](https://yichaozhou.com), [Haozhi Qi](http://haozhi.io), [Yi Ma](https://people.eecs.berkeley.edu/~yima/). ["End-to-End Wireframe Parsing."](https://arxiv.org/abs/1905.03246)  ICCV 2019*.

## Introduction

[L-CNN](https://arxiv.org/abs/1905.03246) is a conceptually simple yet effective neural network for detecting the wireframe from a given image. It outperforms the previous state-of-the-art wireframe and line detectors by a large margin. We hope that this repository serves as an easily reproducible baseline for future researches in this area.


## Code Structure


This implementation is largely based on [LCNN](https://github.com/zhou13/lcnn).  (Thanks Yichao Zhou for such a nice implementation!)


Below is a quick overview of the function of each file.

```bash
########################### Data ###########################
figs/
data/                           # default folder for placing the data
    dpid/                       # folder for Digitize PID dataset (Paliwal et al.)
logs/                           # default folder for storing the output during training
########################### Code ###########################
config/                         # neural network hyper-parameters and configurations
    dpid.yaml              # default parameter for ShanghaiTech dataset
dataset/                        # all scripts related to data generation
    wireframe.py                # script for pre-processing the ShanghaiTech dataset to npz
Evaluation/
    Post_Processing/            # post processing scripts
        post_keypoints.py       # script for post processing keypoints
        post_lines.py           # script for post processing lines
    mAP_keypoint.py             # script for evaluating mAP for keypoint detection
    mAP_keypoints_seperate_classes.py # script for evaluating mAP for symbol detection per class
    sAP_lines.py                # script for evaluating sAP for line detection
    confusion_matrix.py         # script for generating confusion matrix
lcnn/                           # lcnn module so you can "import lcnn" in other scripts
    models/                     # neural network structure
        hourglass_pose.py       # backbone network (stacked hourglass)
        line_vectorizer.py      # sampler and line verification network
        multitask_learner.py    # network for multi-task learning
    datasets.py                 # reading the training data
    metrics.py                  # functions for evaluation metrics
    trainer.py                  # trainer
    config.py                   # global variables for configuration
    utils.py                    # misc functions
demo.py                         # script for detecting wireframes for an image
train.py                        # script for training the neural network
post.py                         # script for post-processing
process.py                      # script for processing a dataset from a checkpoint
```

## Reproducing Results


### Training
To train the neural network on GPU 0 (specified by `-d 0`) with the default parameters, execute
```bash
python ./train.py -d 0 --identifier baseline config/dpid.yaml
```

## Testing Pretrained Models
To generate wireframes on the validation dataset with the pretrained model, execute

```bash
./process.py config/dpid.yaml <path-to-checkpoint.pth> data/dpid logs/pretrained-model/npz/000312000
```

### Post Processing

To post process the outputs from neural network use the scripts under post_processing folder.

### Evaluation

To evaluate the performance of the model, use the scripts under evaluation folder. 
