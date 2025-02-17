from torch.autograd import Variable
from torch.utils.data import DataLoader
from modules import SOFVSR
from data_utils import TrainsetLoader, OFR_loss, ValidationsetLoader
import torch.backends.cudnn as cudnn
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degradation", type=str, default='BI')
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_iters', type=int, default=200000, help='number of iterations to train')
    parser.add_argument('--trainset_dir', type=str, default='data/train')
    parser.add_argument('--version', type=str, default='sof') # mSOF-VSR이 변화준 모델
    parser.add_argument('--valset_dir', type=str, default='data/val')
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--hevc_step', type=int, default=2)
    return parser.parse_args()


def main(cfg):
    # model
    net = SOFVSR(cfg, is_training=True)
    if cfg.gpu_mode:
        net.cuda()
    cudnn.benchmark = True

    # dataloader
    train_set = TrainsetLoader(cfg)
    train_loader = DataLoader(train_set, num_workers=11, batch_size=cfg.batch_size, shuffle=True)
    val_set = ValidationsetLoader(cfg)
    val_loader = DataLoader(val_set, num_workers=2, batch_size=64, shuffle=True)

    # train
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    milestones = [80000, 160000] # 80k마다 learning rate * gamma를 곱한다.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    criterion = torch.nn.MSELoss()
    loss_list = []

    for idx_iter, (LR, HR) in enumerate(train_loader):
        scheduler.step()
        net.train()

        # data
        b, n_frames, h_lr, w_lr = LR.size() # 배치, frame수, 높이, 넓이로 나옴
        idx_center = (n_frames - 1) // 2

        LR, HR = Variable(LR), Variable(HR)
        if cfg.gpu_mode:
            LR = LR.cuda()
            HR = HR.cuda()
        LR = LR.view(b, -1, 1, h_lr, w_lr) # batch, frame 갯수, channel수, h, w로 reshape하는 과정
        HR = HR.view(b, -1, 1, h_lr * cfg.scale, w_lr * cfg.scale)
        # print('before net lr: ', LR.shape)

        # inference
        #inference는 16:9로 하는데 loss 구할때는 12:9로 들어가도록 코드 짜야함
        flow_L1, flow_L2, flow_L3, SR = net(LR)


        # loss
        loss_SR = criterion(SR, HR[:, idx_center, :, :, :])
        loss_OFR = torch.zeros(1).cuda()


        for i in range(n_frames):
            if i != idx_center:
                loss_L1 = OFR_loss(F.avg_pool2d(LR[:, i, :, :, :], kernel_size=2),
                                   F.avg_pool2d(LR[:, idx_center, :, :, :], kernel_size=2),
                                   flow_L1[i])
                loss_L2 = OFR_loss(LR[:, i, :, :, :], LR[:, idx_center, :, :, :], flow_L2[i])
                loss_L3 = OFR_loss(HR[:, i, :, :, :], HR[:, idx_center, :, :, :], flow_L3[i])
                loss_OFR = loss_OFR + loss_L3 + 0.2 * loss_L2 + 0.1 * loss_L1


        loss = loss_SR + 0.01 * loss_OFR / (n_frames - 1)
        # print("loss: ", loss)
        loss_list.append(loss.data.cpu())

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save checkpoint
        if idx_iter % 1000 ==0 or idx_iter == 199999:
            print('Iteration---%6d,   loss---%f' % (idx_iter + 1, np.array(loss_list).mean()))
            if cfg.version == 'msof' and cfg.hevc_step:
                save_path = 'log/msof/'+ str(cfg.hevc_step)+'/' + cfg.degradation + '_x' + str(cfg.scale)
            else:
                save_path = 'log/sof/'+str(cfg.hevc_step)+'/' + cfg.degradation + '_x' + str(cfg.scale)
            save_name = cfg.degradation + '_x' + str(cfg.scale) + '_iter' + str(idx_iter) + '.pth'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(net.state_dict(), save_path + '/' + save_name)


        if idx_iter % 1000 == 0:
            val_loss_list = []
            net.eval()
            with torch.no_grad():
                for idx, (lr, hr) in enumerate(val_loader):
                    b, n_frames, h_lr, w_lr = lr.size()  # 배치, frame수, 높이, 넓이로 나옴
                    # print(lr.size())
                    idx_center = (n_frames - 1) // 2

                    lr, hr = Variable(lr), Variable(hr)
                    if cfg.gpu_mode:
                        lr = lr.cuda()
                        hr = hr.cuda()
                    lr = lr.view(b, -1, 1, h_lr, w_lr)  # batch, frame 갯수, channel수, h, w로 reshape하는 과정
                    hr = hr.view(b, -1, 1, h_lr * cfg.scale, w_lr * cfg.scale)
                    flow_l1, flow_l2, flow_l3, sr = net(lr)
                    # print('net out')

                    # loss
                    loss_sr = criterion(sr, hr[:, idx_center, :, :, :])
                    loss_val = torch.zeros(1).cuda()

                    for i in range(n_frames):
                        if i != idx_center:
                            loss_l1 = OFR_loss(F.avg_pool2d(lr[:, i, :, :, :], kernel_size=2),
                                               F.avg_pool2d(lr[:, idx_center, :, :, :], kernel_size=2),
                                               flow_l1[i])
                            loss_l2 = OFR_loss(lr[:, i, :, :, :], lr[:, idx_center, :, :, :], flow_l2[i])
                            loss_l3 = OFR_loss(hr[:, i, :, :, :], hr[:, idx_center, :, :, :], flow_l3[i])
                            loss_val = loss_val + loss_l3 + 0.2 * loss_l2 + 0.1 * loss_l1

                    loss = loss_sr + 0.01 * loss_val / (n_frames - 1)
                    val_loss_list.append(loss.data.cpu())
                print('%d val loss----: %f' % (idx_iter+1, np.array(val_loss_list).mean()))



if __name__ == '__main__':
    cfg = parse_args()
    torch.cuda.set_device(cfg.gpu_num)
    print(torch.cuda.current_device())
    main(cfg)
