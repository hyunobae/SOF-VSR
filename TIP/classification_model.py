import os
import random
import torch.nn.functional as F
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset


def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img


class TrainsetDataloader(Dataset):
    def __init__(self, cfg):
        super(TrainsetDataloader, self).__init__()
        self.trainset_dir = cfg.trainset_dir
        self.batch_size = cfg.batch_size
        self.video_list = os.listdir(cfg.trainset_dir)
        self.label_dir = cfg.trainset_dir
        self.n_iters = cfg.epoch * cfg.batch_size


    def __getitem__(self, idx):
        idx_video = random.randint(0, len(self.video_list) - 1)

        img_dir = self.trainset_dir + '/' + self.video_list[idx_video] + '/hr'

        # 정답 label 가져와야함
        label = self.label_dir


        t0 = Image.open(img_dir +'/hr' + str(idx)+'.png')
        t1 = Image.open(img_dir +'/hr' + str(idx+1)+'.png')
        t2 = Image.open(img_dir +'/hr' + str(idx+2)+'.png')

        t0 = np.array(t0, dtype=np.float32) / 255.0
        t1 = np.array(t1, dtype=np.float32) / 255.0
        t2 = np.array(t2, dtype=np.float32) / 255.0

        df0 = t1 - t0
        df1 = t1 - t2

        diff = (df0 - df1)**2

        return toTensor(diff), label

    def __len__(self):
        return self.n_iters


class TestsetDataloader(Dataset):
    def __init__(self, cfg):






class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            #input: 960x540
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # 30, 16
        self.dense = nn.Sequential(
            nn.Linear(512 * 30 *16, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x)
        output = F.softmax(x)
        return output


