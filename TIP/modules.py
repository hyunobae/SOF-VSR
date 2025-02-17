import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class SOFVSR(nn.Module):
    def __init__(self, cfg, n_frames=3, is_training=True):
        super(SOFVSR, self).__init__()
        self.scale = cfg.scale
        self.is_training = is_training
        self.OFR = OFRnet(scale=cfg.scale, channels=320)
        self.SR = SRnet(scale=cfg.scale, channels=320, n_frames=n_frames)

    def forward(self, x):
        b, n_frames, c, h, w = x.size()     # x: b*n*c*h*w
        idx_center = (n_frames - 1) // 2

        # batch, input 갯수, channel, h, w = x.size()
        # motion estimation
        flow_L1 = []
        flow_L2 = []
        flow_L3 = []
        input = []

        for idx_frame in range(n_frames):
            if idx_frame != idx_center: # t-1과 center frame, center frame과 t+1을 각각 input으로 만들어준다
                input.append(torch.cat((x[:,idx_frame,:,:,:], x[:,idx_center,:,:,:]), 1))
        optical_flow_L1, optical_flow_L2, optical_flow_L3 = self.OFR(torch.cat(input, 0))

        optical_flow_L1 = optical_flow_L1.view(-1, b, 2, h//2, w//2)
        optical_flow_L2 = optical_flow_L2.view(-1, b, 2, h, w)
        optical_flow_L3 = optical_flow_L3.view(-1, b, 2, h*self.scale, w*self.scale)

        # motion compensation 이해해야함
        draft_cube = []
        draft_cube.append(x[:, idx_center, :, :, :])# b, n, c, h, w  x=(32x32)

        for idx_frame in range(n_frames):
            if idx_frame == idx_center:
                flow_L1.append([])
                flow_L2.append([])
                flow_L3.append([])
            if idx_frame != idx_center:
                if idx_frame < idx_center:
                    idx = idx_frame
                if idx_frame > idx_center:
                    idx = idx_frame - 1

                flow_L1.append(optical_flow_L1[idx, :, :, :, :])# idx, b, c, h, w
                flow_L2.append(optical_flow_L2[idx, :, :, :, :])
                flow_L3.append(optical_flow_L3[idx, :, :, :, :])

                # optical flow l3 / scale factor : space to depth transformation 수행하며 사이즈가 다시 작아지니 optical flow의 값도 변경함
                for i in range(self.scale):# 4배
                    for j in range(self.scale):
                        draft = optical_flow_warp(x[:, idx_frame, :, :, :],
                                                  optical_flow_L3[idx, :, :, i::self.scale, j::self.scale] / self.scale)
                        #4칸씩 이동하며 가져온다
                        draft_cube.append(draft) #x와 같은 size의 draft cube -> 32개의 draft cube 생성됨 + center frame 1 = 33 (list 33) drafe=(4,1,32,32)
        draft_cube = torch.cat(draft_cube, 1)# (4, 33, 32, 32)

        # super-resolution
        SR = self.SR(draft_cube)

        if self.is_training:
            return flow_L1, flow_L2, flow_L3, SR
        if not self.is_training:
            return SR


class OFRnet(nn.Module):
    def __init__(self, scale, channels):
        super(OFRnet, self).__init__()
        self.pool = nn.AvgPool2d(2)
        self.scale = scale

        ## RNN part
        self.RNN1 = nn.Sequential(
            nn.Conv2d(4, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            CasResB(3, channels)
        )
        self.RNN2 = nn.Sequential(
            nn.Conv2d(channels, 2, 3, 1, 1, bias=False),
        )

        # SR part
        SR = []
        SR.append(CasResB(3, channels))
        if self.scale == 4:
            SR.append(nn.Conv2d(channels, 64 * 4, 1, 1, 0, bias=False))
            SR.append(nn.PixelShuffle(2))
            SR.append(nn.LeakyReLU(0.1, inplace=True))
            SR.append(nn.Conv2d(64, 64 * 4, 1, 1, 0, bias=False))
            SR.append(nn.PixelShuffle(2))
            SR.append(nn.LeakyReLU(0.1, inplace=True))
        elif self.scale == 3:
            SR.append(nn.Conv2d(channels, 64 * 9, 1, 1, 0, bias=False))
            SR.append(nn.PixelShuffle(3))
            SR.append(nn.LeakyReLU(0.1, inplace=True))
        elif self.scale == 2:
            SR.append(nn.Conv2d(channels, 64 * 4, 1, 1, 0, bias=False))
            SR.append(nn.PixelShuffle(2))
            SR.append(nn.LeakyReLU(0.1, inplace=True))
        SR.append(nn.Conv2d(64, 2, 3, 1, 1, bias=False))

        self.SR = nn.Sequential(*SR)

    def __call__(self, x):                  # x: b*2*h*w
        """
        Interpolate 결과 * 2 는 뭔지 모르겠다..
        frame을 여러개 넣거나, 구조를 바꾸거나, 가중치를 주거나.
        """
        #Part 1
        x_L1 = self.pool(x)
        b, c, h, w = x_L1.size() # input pair가 2개여서 2인 듯 하다.
        input_L1 = torch.cat((x_L1, torch.zeros(b, 2, h, w).cuda()), 1) # x_L1이랑 차원 맞추려고 0으로 채운 optical flow랑 cat
        optical_flow_L1 = self.RNN2(self.RNN1(input_L1))
        # flow를 2배 interpolation -> flow 값도 2배 증가해야함. 따라서, *2해줌
        optical_flow_L1_upscaled = F.interpolate(optical_flow_L1, scale_factor=2, mode='bilinear', align_corners=False) * 2

        #Part 2
        # t-1 + center frame LR과 optical flow warp
        x_L2 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], 1), optical_flow_L1_upscaled)
        input_L2 = torch.cat((x_L2, torch.unsqueeze(x[:, 1, :, :], 1), optical_flow_L1_upscaled), 1)
        optical_flow_L2 = self.RNN2(self.RNN1(input_L2)) + optical_flow_L1_upscaled
        op2_size = optical_flow_L2.shape

        #Part 3
        x_L3 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], 1), optical_flow_L2)
        input_L3 = torch.cat((x_L3, torch.unsqueeze(x[:, 1, :, :], 1), optical_flow_L2), 1)
        optical_flow_L3 = self.SR(self.RNN1(input_L3)) + \
                          F.interpolate(optical_flow_L2, scale_factor=self.scale, mode='bilinear', align_corners=False) * self.scale
        return optical_flow_L1, optical_flow_L2, optical_flow_L3


class SRnet(nn.Module):
    def __init__(self, scale, channels, n_frames):
        super(SRnet, self).__init__()
        body = []
        body.append(nn.Conv2d(1 * scale ** 2 * (n_frames-1) + 1, channels, 3, 1, 1, bias=False))
        body.append(nn.LeakyReLU(0.1, inplace=True))
        body.append(CasResB(8, channels))
        if scale == 4:
            body.append(nn.Conv2d(channels, 64 * 4, 1, 1, 0, bias=False))
            body.append(nn.PixelShuffle(2)) # h, w 32 -> 64
            body.append(nn.LeakyReLU(0.1, inplace=True))
            body.append(nn.Conv2d(64, 64 * 4, 1, 1, 0, bias=False)) # feature map 수가 64 (h, w와 상관 x)
            body.append(nn.PixelShuffle(2))# 64 -> 128
            body.append(nn.LeakyReLU(0.1, inplace=True))
        elif scale == 3:
            body.append(nn.Conv2d(channels, 64 * 9, 1, 1, 0, bias=False))
            body.append(nn.PixelShuffle(3))
            body.append(nn.LeakyReLU(0.1, inplace=True))
        elif scale == 2:
            body.append(nn.Conv2d(channels, 64 * 4, 1, 1, 0, bias=False))
            body.append(nn.PixelShuffle(2))
            body.append(nn.LeakyReLU(0.1, inplace=True))
        body.append(nn.Conv2d(64, 1, 3, 1, 1, bias=True))

        self.body = nn.Sequential(*body)

    def __call__(self, x):
        out = self.body(x)
        return out


class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels//2, channels//2, 1, 1, 0, bias=False), # channel split해서 원본의 절반만 있다.
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels//2, channels//2, 3, 1, 1, bias=False, groups=channels//2),
            nn.Conv2d(channels // 2, channels // 2, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        input = x[:, x.shape[1]//2:, :, :]
        out = torch.cat((x[:, :x.shape[1]//2, :, :], self.body(input)), 1)
        return channel_shuffle(out, 2)


class CasResB(nn.Module):
    def __init__(self, n_ResB, channels):
        super(CasResB, self).__init__()
        body = []
        for i in range(n_ResB):
            body.append(ResB(channels))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    x = x.view(b, groups, c//groups,  h, w)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.view(b, -1, h, w)
    return x


def optical_flow_warp(image, image_optical_flow):
    """
    Arguments
        image_ref: reference images tensor, (b, c, h, w)
        image_optical_flow: optical flow to image_ref (b, 2, h, w)
    """
    b, _ , h, w = image.size()
    grid = np.meshgrid(range(w), range(h))
    grid = np.stack(grid, axis=-1).astype(np.float64) # (32, 32, 2) -> (x, y)로 나타냄

    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1 #w축
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1 #h축
    grid = grid.transpose(2, 0, 1) # (32, 32, 2) -> (2, 32, 32)
    grid = np.tile(grid, (b, 1, 1, 1)) # (2, 32, 32) -> (128, 2, 32, 32)
    grid = Variable(torch.Tensor(grid))
    if image_optical_flow.is_cuda == True:
        grid = grid.cuda()

    flow_0 = torch.unsqueeze(image_optical_flow[:, 0, :, :] * 31 / (w - 1), dim=1)
    flow_1 = torch.unsqueeze(image_optical_flow[:, 1, :, :] * 31 / (h - 1), dim=1) # flow field의 범위 때문에 normalization하는 듯
    grid = grid + torch.cat((flow_0, flow_1),1) # 생성한 grid에 optical flow를 모아준다. -> warping
    grid = grid.transpose(1, 2) # (128, 2, 32, 32) -> (128, 32, 2, 32)
    grid = grid.transpose(3, 2) # (128, 32, 2, 32) -> (128, 32, 32, 2)

    output = F.grid_sample(image, grid, padding_mode='border')#warping해서 compensate된 새로운 image 생성해줌
    return output
