from PIL import Image
from torch.utils.data.dataset import Dataset
from modules import optical_flow_warp
import numpy as np
import os
import torch
import random


def get_hevc_idx(idx_frame, step):
    left_frame = 0
    right_frame = 0

    i = idx_frame // 8  # I frame idx이자 그냥 idx 역할
    p = idx_frame % 8

    left_I = 0 + i * 8
    right_I = left_I + 8

    is_b = idx_frame % 2  # 나머지 1이면 무조건 b frame -> 무조건 -1, +1 보면 됨

    if step == 2:  # 2, 4, 8로 나누면 될듯 -> optical flow가 예민하기 때문에 16은 무리
        if is_b == 1:  # b frame인 경우
            left_frame = idx_frame - 1
            right_frame = idx_frame + 1

        elif is_b == 0:
            left_frame = idx_frame - 2
            right_frame = idx_frame + 2

    elif step == 4:  # 최대 볼 수 있는 거리가 4라고 가정하자 -> 거리 4 내에 있는 가장 좋은 frame 참고하는 방식으로
        if p < 4:
            left_frame = left_I
            right_frame = left_I + 4

        elif p == 0 or p == 4:
            left_frame = left_I
            right_frame = right_I

        else:
            left_frame = right_I - 4
            right_frame = right_I

    return left_frame, right_frame


def get_msof_idx(idx_frame):
    left_frame = 0
    right_frame = 0

    # left I frame, right I frame fix
    n = idx_frame // 16  # I frame idx
    left_I = 0 + n * 16
    right_I = left_I + 16

    p = (idx_frame // 5) + 1  # 0이면 0~4는 5를 참고 -> 몇번째 p block인지 의미함

    # p frame의 idx
    left_p = left_I + 5  # I다음 나오는 첫번째 P idx, P의 idx
    mid_p = left_p + 5
    right_p = right_I - 1

    # 현재 frame이 I, P, B 구분
    if idx_frame == 0:
        left_frame = left_I
        right_frame = left_p

    elif idx_frame % 16 == 0:  # if cur frame is I frame
        left_frame = idx_frame - 1  # 바로 옆 p frame
        right_frame = idx_frame + 3  # B 중 좋은 frame

    # if cur frame is P frame
    elif idx_frame == left_p or idx_frame == mid_p or idx_frame == right_p:  # if cur frame is P frame
        if idx_frame == left_p:
            left_frame = idx_frame - 2  # t-2의 B frame
            right_frame = mid_p

        elif idx_frame == mid_p:
            left_frame = idx_frame - 2
            right_frame = right_p

        elif idx_frame == right_p:
            left_frame = idx_frame - 2
            right_frame = right_I

    else:  # if cur frame is neither I frame nor P frame -> B
        # (p-1)은 I와 I 사이에 몇번째 p block인지 판단함 -> p block은 bbbbp를 의미
        if (p - 1) % 3 == 0:  # cur frame의 좌측 I frame에 인접한 B
            if idx_frame == (left_p - 1):  # BBB B P에서 ^B^인 경우
                left_frame = idx_frame - 1
                right_frame = left_p

            elif idx_frame == left_p - 2:  # p frame -2 는 항상 좋은 B frame이다.
                left_frame = left_I
                right_frame = left_p

            else:
                left_frame = left_I
                right_frame = left_p - 2

        elif (p - 1) % 3 == 2:  # cur frame의 우측에 I frame이 위치할때
            if idx_frame == right_p - 1:
                left_frame = idx_frame - 1
                right_frame = right_I

            elif idx_frame == right_p - 2:
                left_frame = mid_p
                right_frame = right_I

            else:
                left_frame = mid_p
                right_frame = right_p - 2

        else:  # 중간 p block인 경우
            if idx_frame == mid_p - 1:
                left_frame = idx_frame - 1
                right_frame = mid_p

            elif idx_frame == mid_p - 2:
                left_frame = left_p
                right_frame = mid_p

            else:
                left_frame = left_p
                right_frame = mid_p - 2

    return left_frame, right_frame


class TrainsetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainsetLoader).__init__()
        self.trainset_dir = cfg.trainset_dir
        self.scale = cfg.scale
        self.patch_size = cfg.patch_size
        self.n_iters = cfg.n_iters * cfg.batch_size
        self.video_list = os.listdir(cfg.trainset_dir)
        self.degradation = cfg.degradation
        self.version = cfg.version
        self.step = cfg.hevc_step

    def __getitem__(self, idx):
        if self.version == 'sof':
            idx_video = random.randint(0, len(self.video_list) - 1)
            # idx_frame = random.randint(0, 14)  # #frames of training videos is 31, 31-3=28   test로 17장만 사용해본다.
            idx_frame = random.randint(0, 30)  # lr0~lr32만 참고한다.
            lr_dir = self.trainset_dir + '/' + self.video_list[idx_video] + '/lr_x' + str(
                self.scale) + '_' + self.degradation
            hr_dir = self.trainset_dir + '/' + self.video_list[idx_video] + '/hr'

            # read HR & LR frames
            LR0 = Image.open(lr_dir + '/lr' + str(idx_frame) + '.png')
            LR1 = Image.open(lr_dir + '/lr' + str(idx_frame + 1) + '.png')
            LR2 = Image.open(lr_dir + '/lr' + str(idx_frame + 2) + '.png')
            HR0 = Image.open(hr_dir + '/hr' + str(idx_frame) + '.png')
            HR1 = Image.open(hr_dir + '/hr' + str(idx_frame + 1) + '.png')
            HR2 = Image.open(hr_dir + '/hr' + str(idx_frame + 2) + '.png')


        elif self.version == 'msof':
            idx_video = random.randint(0, len(self.video_list) - 1)
            idx_frame = random.randint(2, 31)
            lr_dir = self.trainset_dir + '/' + self.video_list[idx_video] + '/lr_x' + str(
                self.scale) + '_' + self.degradation
            hr_dir = self.trainset_dir + '/' + self.video_list[idx_video] + '/hr'

            left_frame, right_frame = get_hevc_idx(idx_frame, self.step)

            # 중간 frame sr을 위해 양쪽 I frame을 참조한다.
            LR0 = Image.open(lr_dir + '/lr' + str(left_frame) + '.png')
            LR1 = Image.open(lr_dir + '/lr' + str(idx_frame) + '.png')
            LR2 = Image.open(lr_dir + '/lr' + str(right_frame) + '.png')
            HR0 = Image.open(hr_dir + '/hr' + str(left_frame) + '.png')
            HR1 = Image.open(hr_dir + '/hr' + str(idx_frame) + '.png')
            HR2 = Image.open(hr_dir + '/hr' + str(right_frame) + '.png')

        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        LR1 = np.array(LR1, dtype=np.float32) / 255.0
        LR2 = np.array(LR2, dtype=np.float32) / 255.0
        HR0 = np.array(HR0, dtype=np.float32) / 255.0
        HR1 = np.array(HR1, dtype=np.float32) / 255.0
        HR2 = np.array(HR2, dtype=np.float32) / 255.0

        # extract Y channel for LR inputs
        HR0 = rgb2y(HR0)
        HR1 = rgb2y(HR1)
        HR2 = rgb2y(HR2)
        LR0 = rgb2y(LR0)
        LR1 = rgb2y(LR1)
        LR2 = rgb2y(LR2)

        # crop patchs randomly
        HR0, HR1, HR2, LR0, LR1, LR2 = random_crop(HR0, HR1, HR2, LR0, LR1, LR2, self.patch_size, self.scale)

        HR0 = HR0[:, :, np.newaxis]
        HR1 = HR1[:, :, np.newaxis]
        HR2 = HR2[:, :, np.newaxis]
        LR0 = LR0[:, :, np.newaxis]
        LR1 = LR1[:, :, np.newaxis]
        LR2 = LR2[:, :, np.newaxis]

        HR = np.concatenate((HR0, HR1, HR2), axis=2)
        LR = np.concatenate((LR0, LR1, LR2), axis=2)

        # data augmentation
        LR, HR = augmentation()(LR, HR)

        return toTensor(LR), toTensor(HR)

    def __len__(self):
        return self.n_iters


class TestsetLoader(Dataset):
    def __init__(self, cfg, video_name):
        super(TestsetLoader).__init__()
        self.dataset_dir = cfg.testset_dir + '/' + video_name
        self.degradation = cfg.degradation
        self.scale = cfg.scale
        self.frame_list = os.listdir(self.dataset_dir + '/lr_x' + str(self.scale) + '_' + self.degradation)
        self.version = cfg.version
        self.step = cfg.hevc_step

    def __getitem__(self, idx):
        dir = self.dataset_dir + '/lr_x' + str(self.scale) + '_' + self.degradation
        idx = idx + 2
        if self.version == 'sof':
            LR0 = Image.open(dir + '/' + 'lr' + str(idx) + '.png')
            LR1 = Image.open(dir + '/' + 'lr' + str(idx + 1) + '.png')
            LR2 = Image.open(dir + '/' + 'lr' + str(idx + 2) + '.png')

        elif self.version == 'msof':
            left_frame, right_frame = get_hevc_idx(idx, self.step)

            LR0 = Image.open(dir + '/' + 'lr' + str(left_frame) + '.png')
            LR1 = Image.open(dir + '/' + 'lr' + str(idx + 1) + '.png')
            LR2 = Image.open(dir + '/' + 'lr' + str(right_frame) + '.png')

        W, H = LR1.size

        # H and W should be divisible by 2
        W = int(W // 2) * 2
        H = int(H // 2) * 2
        LR0 = LR0.crop([0, 0, W, H])
        LR1 = LR1.crop([0, 0, W, H])
        LR2 = LR2.crop([0, 0, W, H])

        LR1_bicubic = LR1.resize((round(W * self.scale * (4 / 3)), H * self.scale), Image.BICUBIC)

        LR1 = LR1.resize((round(W * (4 / 3)), H), Image.BICUBIC)
        LR0 = LR0.resize((round(W * (4 / 3)), H), Image.BICUBIC)
        LR2 = LR2.resize((round(W * (4 / 3)), H), Image.BICUBIC)

        LR1_bicubic = np.array(LR1_bicubic, dtype=np.float32) / 255.0

        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        LR1 = np.array(LR1, dtype=np.float32) / 255.0
        LR2 = np.array(LR2, dtype=np.float32) / 255.0

        # extract Y channel for LR inputs
        LR0_y, _, _ = rgb2ycbcr(LR0)
        LR1_y, _, _ = rgb2ycbcr(LR1)
        LR2_y, _, _ = rgb2ycbcr(LR2)

        LR0_y = LR0_y[:, :, np.newaxis]
        LR1_y = LR1_y[:, :, np.newaxis]
        LR2_y = LR2_y[:, :, np.newaxis]
        LR = np.concatenate((LR0_y, LR1_y, LR2_y), axis=2)

        LR = toTensor(LR)

        # generate Cr, Cb channels using bicubic interpolation
        _, SR_cb, SR_cr = rgb2ycbcr(LR1_bicubic)

        return LR, SR_cb, SR_cr

    def __len__(self):
        # return len(self.frame_list) - 2
        return 29


class ValidationsetLoader(Dataset):
    def __init__(self, cfg):
        super(ValidationsetLoader).__init__()
        self.dataset_dir = cfg.valset_dir
        self.degradation = cfg.degradation
        self.scale = cfg.scale
        self.video_list = os.listdir(self.dataset_dir)
        self.version = cfg.version
        self.patch_size = cfg.patch_size
        self.version = cfg.version
        self.step = cfg.hevc_step
        self.n_iters = cfg.n_iters * cfg.batch_size
        self.batch_size = cfg.batch_size

    def __getitem__(self, idx_frame):
        idx_video = random.randint(0, len(self.video_list) - 1)
        if self.version == 'sof':
            idx_frame = random.randint(1, 31)  # lr0~lr16만 참고한다.

        elif self.version == 'msof':
            idx_frame = random.randint(2, 30)

        lr_dir = self.dataset_dir + '/' + self.video_list[idx_video] + '/lr_x' + str(
            self.scale) + '_' + self.degradation
        hr_dir = self.dataset_dir + '/' + self.video_list[idx_video] + '/hr'

        if self.version == 'msof':
            left_frame, right_frame = get_hevc_idx(idx_frame, self.step)

            # 중간 frame sr을 위해 양쪽 I frame을 참조한다.
            LR0 = Image.open(lr_dir + '/lr' + str(left_frame) + '.png')
            LR1 = Image.open(lr_dir + '/lr' + str(idx_frame) + '.png')
            LR2 = Image.open(lr_dir + '/lr' + str(right_frame) + '.png')
            HR0 = Image.open(hr_dir + '/hr' + str(left_frame) + '.png')
            HR1 = Image.open(hr_dir + '/hr' + str(idx_frame) + '.png')
            HR2 = Image.open(hr_dir + '/hr' + str(right_frame) + '.png')

        elif self.version == 'sof':
            LR0 = Image.open(lr_dir + '/lr' + str(idx_frame) + '.png')
            LR1 = Image.open(lr_dir + '/lr' + str(idx_frame + 1) + '.png')
            LR2 = Image.open(lr_dir + '/lr' + str(idx_frame + 2) + '.png')
            HR0 = Image.open(hr_dir + '/hr' + str(idx_frame) + '.png')
            HR1 = Image.open(hr_dir + '/hr' + str(idx_frame + 1) + '.png')
            HR2 = Image.open(hr_dir + '/hr' + str(idx_frame + 2) + '.png')


        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        LR1 = np.array(LR1, dtype=np.float32) / 255.0
        LR2 = np.array(LR2, dtype=np.float32) / 255.0
        HR0 = np.array(HR0, dtype=np.float32) / 255.0
        HR1 = np.array(HR1, dtype=np.float32) / 255.0
        HR2 = np.array(HR2, dtype=np.float32) / 255.0

        # extract Y channel for LR inputs
        HR0 = rgb2y(HR0)
        HR1 = rgb2y(HR1)
        HR2 = rgb2y(HR2)
        LR0 = rgb2y(LR0)
        LR1 = rgb2y(LR1)
        LR2 = rgb2y(LR2)

        # crop patchs randomly
        HR0, HR1, HR2, LR0, LR1, LR2 = random_crop(HR0, HR1, HR2, LR0, LR1, LR2, self.patch_size, self.scale)

        HR0 = HR0[:, :, np.newaxis]
        HR1 = HR1[:, :, np.newaxis]
        HR2 = HR2[:, :, np.newaxis]
        LR0 = LR0[:, :, np.newaxis]
        LR1 = LR1[:, :, np.newaxis]
        LR2 = LR2[:, :, np.newaxis]

        HR = np.concatenate((HR0, HR1, HR2), axis=2)
        LR = np.concatenate((LR0, LR1, LR2), axis=2)

        # data augmentation
        # LR, HR = augmentation()(LR, HR)

        LR = np.ascontiguousarray(LR)
        HR = np.ascontiguousarray(HR)

        return toTensor(LR), toTensor(HR)

    def __len__(self):
        return self.batch_size #batchsize * iters


class augmentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:
            input = input[:, ::-1, :]
            target = target[:, ::-1, :]
        if random.random() < 0.5:
            input = input[::-1, :, :]
            target = target[::-1, :, :]
        if random.random() < 0.5:
            input = input.transpose(1, 0, 2)
            target = target.transpose(1, 0, 2)
        return np.ascontiguousarray(input), np.ascontiguousarray(target)


def random_crop(HR0, HR1, HR2, LR0, LR1, LR2, patch_size_lr, scale):
    """
    HR과 LR이 서로 같은 부위를 각각 128x128, 32x32의 patch를 가지도록 한다.
    결국 patch를 HR하는 것을 학습하는 것 같다.
    """
    h_hr, w_hr = HR0.shape
    h_lr = h_hr // scale
    w_lr = w_hr // scale
    idx_h = random.randint(10, h_lr - patch_size_lr - 10)
    idx_w = random.randint(10, w_lr - patch_size_lr - 10)

    h_start_hr = (idx_h - 1) * scale
    h_end_hr = (idx_h - 1 + patch_size_lr) * scale
    w_start_hr = (idx_w - 1) * scale
    w_end_hr = (idx_w - 1 + patch_size_lr) * scale

    h_start_lr = idx_h - 1
    h_end_lr = idx_h - 1 + patch_size_lr
    w_start_lr = idx_w - 1
    w_end_lr = idx_w - 1 + patch_size_lr

    HR0 = HR0[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    HR1 = HR1[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    HR2 = HR2[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    LR0 = LR0[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    LR1 = LR1[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    LR2 = LR2[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    return HR0, HR1, HR2, LR0, LR1, LR2


def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img


def rgb2ycbcr(img_rgb):
    ## the range of img_rgb should be (0, 1)
    img_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] + 16 / 255.0
    img_cb = -0.148 * img_rgb[:, :, 0] - 0.291 * img_rgb[:, :, 1] + 0.439 * img_rgb[:, :, 2] + 128 / 255.0
    img_cr = 0.439 * img_rgb[:, :, 0] - 0.368 * img_rgb[:, :, 1] - 0.071 * img_rgb[:, :, 2] + 128 / 255.0
    return img_y, img_cb, img_cr


def ycbcr2rgb(img_ycbcr):
    ## the range of img_ycbcr should be (0, 1)
    img_r = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 1.596 * (img_ycbcr[:, :, 2] - 128 / 255.0)
    img_g = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) - 0.392 * (img_ycbcr[:, :, 1] - 128 / 255.0) - 0.813 * (
            img_ycbcr[:, :, 2] - 128 / 255.0)
    img_b = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 2.017 * (img_ycbcr[:, :, 1] - 128 / 255.0)
    img_r = img_r[:, :, np.newaxis]
    img_g = img_g[:, :, np.newaxis]
    img_b = img_b[:, :, np.newaxis]
    img_rgb = np.concatenate((img_r, img_g, img_b), 2)
    return img_rgb


def rgb2y(img_rgb):
    ## the range of img_rgb should be (0, 1)
    image_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] + 16 / 255.0
    return image_y


def OFR_loss(x0, x1, optical_flow):
    warped = optical_flow_warp(x0, optical_flow)
    loss = torch.mean(torch.abs(x1 - warped)) + 0.1 * L1_regularization(optical_flow)
    return loss


def L1_regularization(image):
    b, _, h, w = image.size()
    reg_x_1 = image[:, :, 0:h - 1, 0:w - 1] - image[:, :, 1:, 0:w - 1]
    reg_y_1 = image[:, :, 0:h - 1, 0:w - 1] - image[:, :, 0:h - 1, 1:]
    reg_L1 = torch.abs(reg_x_1) + torch.abs(reg_y_1)
    return torch.sum(reg_L1) / (b * (h - 1) * (w - 1))
