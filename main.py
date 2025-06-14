from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models.gwcnet import GwcNet
from models.loss import model_loss
from utils import *
from torch.utils.data import DataLoader
import gc
import json
from datetime import datetime
from utils.saver import Saver

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Adaptive Disparity Candidate Prediction Network (ADCPNet)')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--c', type=int, default=1, help='init_channels at feature extraction')
parser.add_argument('--mag', type=int, default=2, help='half of candidate disp num')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels of the 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4, 1], help='growth rate in the 3d network')
parser.add_argument('--channel_CRP', type=int, required=True, help='channnel num of CRP')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')
parser.add_argument('--ckpt_start_epoch', type=int, default=0, help='the epochs at which the program start saving ckpt')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
saver = Saver(args)
print("creating new summary file")
logger = SummaryWriter(saver.experiment_dir)

logfilename = saver.experiment_dir + '/log.txt'

with open(logfilename, 'a') as log:  # wrt running information to log
    log.write('\n\n\n\n')
    log.write('-------------------NEW RUN-------------------\n')
    log.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    log.write('\n')
    json.dump(args.__dict__, log, indent=2)
    log.write('\n')

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = GwcNet(args)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(os.path.join(saver.directory, 'experiment_{}'.format(str(saver.run_id - 1)))) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(saver.directory, 'experiment_{}'.format(str(saver.run_id - 1)), all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))


def train():
    min_EPE = args.maxdisp
    min_D1 = 1
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
            with open(logfilename, 'a') as log:
                log.write('Epoch {}/{}, Iter {}/{}, train loss = {}, time = {:.3f}\n'.format(epoch_idx, args.epochs,
                                                                                                 batch_idx,
                                                                                                 len(TrainImgLoader),
                                                                                                 loss,
                                                                                                 time.time() - start_time))

        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0 and epoch_idx >= args.ckpt_start_epoch:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(saver.experiment_dir, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                 batch_idx,
                                                                                 len(TestImgLoader), loss,
                                                                                 time.time() - start_time))
            with open(logfilename, 'a') as log:
                log.write('Epoch {}/{}, Iter {}/{}, test loss = {}, time = {:.3f}\n'.format(epoch_idx, args.epochs,
                                                                                            batch_idx,
                                                                                            len(TestImgLoader),
                                                                                            loss,
                                                                                            time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        if avg_test_scalars['EPE'][-1] < min_EPE:
            min_EPE = avg_test_scalars['EPE'][-1]
            minEPE_epoch = epoch_idx
        if avg_test_scalars['D1'][-1] < min_D1:
            min_D1 = avg_test_scalars['D1'][-1]
            minD1_epoch = epoch_idx
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        with open(logfilename, 'a') as log:
            js = json.dumps(avg_test_scalars)
            log.write(js)
            log.write('\n')
        gc.collect()
    with open(logfilename, 'a') as log:
        log.write('min_EPE: {}/{}; min_D1: {}/{}'.format(min_EPE, minEPE_epoch, min_D1, minD1_epoch))


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    weights = [0.25, 0.5, 0.5, 1.0]
    loss = []
    weighted_loss = []
    for i in range(len(disp_ests)):
        loss.append(F.smooth_l1_loss(disp_ests[i][mask], disp_gt[mask], size_average=True))
        weighted_loss.append(weights[i] * loss[-1])
    scalar_outputs = {"weighted_loss_sum": sum(weighted_loss)}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    sum(weighted_loss).backward()
    print('sum(weighted_loss): {}'.format(sum(weighted_loss)))
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    disp_ests = model(imgL, imgR)
    if args.dataset != 'kitti':
        for i in range(len(disp_ests)):
            disp_ests[i] = disp_ests[i][:, 4:, :]
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    weights = [0.25, 0.5, 0.5, 1.0]
    loss = []
    weighted_loss = []
    for i in range(len(disp_ests)):
        loss.append(F.smooth_l1_loss(disp_ests[i][mask], disp_gt[mask], size_average=True))
        weighted_loss.append(weights[i] * loss[-1])
    scalar_outputs = {"weighted_loss_sum": sum(weighted_loss)}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    train()
