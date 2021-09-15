import random

from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from data_utils import TrainsetLoader, OFR_loss
import torch.backends.cudnn as cudnn
import argparse
import torch
import numpy as np
import os
import torchvision.transforms as transforms
from classification_model import Network, TrainsetDataloader, TestsetDataloader



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--trainset_dir', type=str, default='data/train')
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--label_dir', type=str, default='data/label')
    parser.add_argument('--epoch', type=int, default=200)
    return parser.parse_args()



def main(cfg):
    model = Network()



if __name__ == '__main__':
    cfg = parse_args()
    torch.cuda.set_device(cfg.gpu_num)
    print(torch.cuda.current_device())
    main(cfg)
