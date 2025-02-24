import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label
from torch.optim import lr_scheduler
from model_2D import *

from config import get_config
from networks.utils import str2bool
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, ThreeStreamBatchSampler,
                                 TwoStreamBatchSampler1)
from networks.net_factory import MCMCCL_Net, net_factory, process_and_project, projection_MLP1, projection_MLP2
from utils import losses as lossess
from utils import ramps, feature_memory, contrastive_losses, val_2d
from networks import archs, losses
from utils.supcon_loss_romdom_seleted import *

from networks.vision_transformer import SwinUnet as ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC',
                    help='Name of Experiment')
parser.add_argument('--exp', type=str, default='MCMCCL', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[224, 224], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=3, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int, default=6, help='multinum of random masks')


parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)

parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])
parser.add_argument('--lr_k', '--learning_rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum_k', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight_decay_k', default=1e-4, type=float,
                    help='weight decay')
parser.add_argument('--nesterov_k', default=False, type=str2bool,
                    help='nesterov')

parser.add_argument('--kan_lr_k', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--kan_weight_decay_k', default=1e-4, type=float,
                    help='weight decay')
parser.add_argument('--scheduler', default='CosineAnnealingLR',
                    choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
parser.add_argument('--min_lr', default=1e-5, type=float,
                    help='minimum learning rate')
parser.add_argument('--factor', default=0.1, type=float)
parser.add_argument('--patience', default=2, type=int)
parser.add_argument('--milestones', default='1,2', type=str)
parser.add_argument('--gamma', default=2 / 3, type=float)
parser.add_argument('--early_stopping', default=-1, type=int,
                    metavar='N', help='early stopping (default: -1)')

args = parser.parse_args()
config = get_config(args)

dice_loss = lossess.DiceLoss(n_classes=4)


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()


def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)
    return probs


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x * 2 / 3), int(img_y * 2 / 3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0

    return mask.long(), loss_mask.long()


def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x * 2 / (3 * shrink_param)), int(img_y * 2 / (3 * shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s * x_split, (x_s + 1) * x_split - patch_x)
            h = np.random.randint(y_s * y_split, (y_s + 1) * y_split - patch_y)
            mask[w:w + patch_x, h:h + patch_y] = 0
            loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    return mask.long(), loss_mask.long()


def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y * 4 / 9)
    h = np.random.randint(0, img_y - patch_y)
    mask[h:h + patch_y, :] = 0
    loss_mask[:, h:h + patch_y, :] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)  # loss = loss_ce
    return loss_dice, loss_ce


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score[ignore != 1] * target[ignore != 1])
        y_sum = torch.sum(target[ignore != 1] * target[ignore != 1])
        z_sum = torch.sum(score[ignore != 1] * score[ignore != 1])
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False, ignore=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def ConstraLoss(inputs, targets):
    m = nn.AdaptiveAvgPool2d(1)
    input_pro = m(inputs)
    input_pro = input_pro.view(inputs.size(0), -1)  # N*C
    targets_pro = m(targets)
    targets_pro = targets_pro.view(targets.size(0), -1)  # N*C
    input_normal = nn.functional.normalize(input_pro, p=2, dim=1)  #  正则化
    targets_normal = nn.functional.normalize(targets_pro, p=2, dim=1)
    res = (input_normal - targets_normal)
    res = res * res
    loss = torch.mean(res)
    return loss


def pre_train_k(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    pre_trained_model = os.path.join(pre_snapshot_path_k, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model = ViT_seg(config, img_size=args.patch_size,
                      num_classes=args.num_classes)
    model.load_from(config)

    model.cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    param_groups = []
    for name, param in model.named_parameters():
        # print(name, "=>", param.shape)
        if 'layer' in name.lower() and 'fc' in name.lower():  # higher lr for kan layers
            # kan_fc_params.append(name)
            param_groups.append({'params': param, 'lr': args.kan_lr_k, 'weight_decay': args.kan_weight_decay_k})
        else:
            # other_params.append(name)
            param_groups.append({'params': param, 'lr': args.lr_k, 'weight_decay': args.weight_decay_k})

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=num_classes)

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            with torch.no_grad():
                img_mask, loss_mask = generate_mask(img_a)
                gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            # -- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            _, out_mixl = model(net_input)
            out_soft_mixl = torch.softmax(out_mixl, dim=1)

            loss_ce = ce_loss(out_mixl, gt_mixl)
            loss_dice = dice_loss(out_soft_mixl, gt_mixl.unsqueeze(1))

            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break

        #scheduler.step()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


def pre_train_u(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    pre_trained_model = os.path.join(pre_snapshot_path_u, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model = MCMCCL_Net(in_chns=1, class_num=num_classes)

    model.cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=num_classes)

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            with torch.no_grad():
                img_mask, loss_mask = generate_mask(img_a)
                gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            # -- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            _, out_mixl = model(net_input)
            out_soft_mixl = torch.softmax(out_mixl, dim=1)

            loss_ce = ce_loss(out_mixl, gt_mixl)
            loss_dice = dice_loss(out_soft_mixl, gt_mixl.unsqueeze(1))

            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path_u, pre_snapshot_path_k, snapshot_path_u, snapshot_path_k):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    pre_trained_model_u = os.path.join(pre_snapshot_path_u, '{}_best_model.pth'.format(args.model))
    pre_trained_model_k = os.path.join(pre_snapshot_path_k, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    model_u = MCMCCL_Net(in_chns=1, class_num=num_classes)
    model_k = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()

    model_u.cuda()
    model_k.cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # Data
    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler1(labeled_idxs, unlabeled_idxs, args.batch_size,
                                           args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    # optimizer
    optimizer_u = optim.SGD(model_u.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_k = optim.SGD(model_k.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0001)

    criterion_dice = DiceLoss(n_classes=4)

    load_net_opt(model_u, optimizer_u, pre_trained_model_u)
    load_net_opt(model_k, optimizer_k, pre_trained_model_k)
    logging.info("Loaded from {}".format(pre_trained_model_u))
    logging.info("Loaded from {}".format(pre_trained_model_k))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model_u.train()
    model_k.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=num_classes)

    proj_mlp1 = projection_MLP1(in_dim=256)
    optimizer_mlp1 = optim.SGD(proj_mlp1.parameters(), lr=0.01,
                              momentum=0.9, weight_decay=0.0001)

    scheduler_mlp1 = lr_scheduler.CosineAnnealingLR(
        optimizer_mlp1, T_max=max_iterations // len(trainloader) + 1, eta_min=args.min_lr)

    proj_mlp2 = projection_MLP2(in_dim=16)
    optimizer_mlp2 = optim.SGD(proj_mlp2.parameters(), lr=0.01,
                               momentum=0.9, weight_decay=0.0001)

    scheduler_mlp2 = lr_scheduler.CosineAnnealingLR(
        optimizer_mlp2, T_max=max_iterations // len(trainloader) + 1, eta_min=args.min_lr)

    proj_mlp1.cuda()
    proj_mlp2.cuda()
    proj_mlp1.train()
    proj_mlp2.train()


    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance_u = 0.0
    best_performance_k = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img = volume_batch[:args.labeled_bs]
            label = label_batch[:args.labeled_bs]
            uimg = volume_batch[args.labeled_bs:args.batch_size]

            # UNet
            feature_k, pseudo_u = model_k(uimg)
            with torch.no_grad():
                pseudo_uu = get_ACDC_masks(pseudo_u, nms=1)
                img_mask_u, loss_mask_u = generate_mask(img)
                y_l2u_u = pseudo_uu * img_mask_u + label * (1 - img_mask_u)
                y_u2l_u = label * img_mask_u + pseudo_uu * (1 - img_mask_u)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            _, pre_fp_u = model_u(uimg, need_fp=True)

            pseudo_u = pseudo_u.detach()
            conf_fp_u = pseudo_u.softmax(dim=1).max(dim=1)[0]
            mask_fp_u = pseudo_u.argmax(dim=1)
            l_fp_u = criterion_dice(pre_fp_u.softmax(dim=1), mask_fp_u.unsqueeze(1).float(),
                                    ignore=(conf_fp_u < 0.95).float())

            x_l2u_u = uimg * img_mask_u + img * (1 - img_mask_u)
            x_u2l_u = img * img_mask_u + uimg * (1 - img_mask_u)

            feature_l2u_u, pre_l2u_u = model_u(x_l2u_u)
            pre_soft_l2u_u = torch.softmax(pre_l2u_u, dim=1)
            feature_u2l_u, pre_u2l_u = model_u(x_u2l_u)
            pre_soft_u2l_u = torch.softmax(pre_u2l_u, dim=1)

            l_l2u_u = 0.5 * (ce_loss(pre_l2u_u, y_l2u_u.long()) + dice_loss(pre_soft_l2u_u, y_l2u_u.unsqueeze(1)))
            l_u2l_u = 0.5 * (ce_loss(pre_u2l_u, y_u2l_u.long()) + dice_loss(pre_soft_u2l_u, y_u2l_u.unsqueeze(1)))

            # KNet
            feature_u, pseudo_k = model_u(uimg)
            with torch.no_grad():
                pseudo_kk = get_ACDC_masks(pseudo_k, nms=1)
                img_mask_k, loss_mask_k = generate_mask(img)
                y_l2u_k = pseudo_kk * img_mask_k + label * (1 - img_mask_k)
                y_u2l_k = label * img_mask_k + pseudo_kk * (1 - img_mask_k)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            _, pre_fp_k = model_k(uimg, need_fp=True)

            pseudo_k = pseudo_k.detach()
            conf_fp_k = pseudo_k.softmax(dim=1).max(dim=1)[0]
            mask_fp_k = pseudo_k.argmax(dim=1)
            l_fp_k = criterion_dice(pre_fp_k.softmax(dim=1), mask_fp_k.unsqueeze(1).float(),
                                    ignore=(conf_fp_k < 0.95).float())

            x_l2u_k = uimg * img_mask_k + img * (1 - img_mask_k)
            x_u2l_k = img * img_mask_k + uimg * (1 - img_mask_k)

            feature_l2u_k, pre_l2u_k = model_k(x_l2u_k)
            pre_soft_l2u_k = torch.softmax(pre_l2u_k, dim=1)
            feature_u2l_k, pre_u2l_k = model_k(x_u2l_k)
            pre_soft_u2l_k = torch.softmax(pre_u2l_k, dim=1)

            l_l2u_k = 0.5 * (ce_loss(pre_l2u_k, y_l2u_k.long()) + dice_loss(pre_soft_l2u_k, y_l2u_k.unsqueeze(1)))
            l_u2l_k = 0.5 * (ce_loss(pre_u2l_k, y_u2l_k.long()) + dice_loss(pre_soft_u2l_k, y_u2l_k.unsqueeze(1)))


            _, _, contrastive_loss1 = process_and_project(feature_u[0], feature_k[0], proj_mlp1)
            contrastive_loss1 = contrastive_loss1.mean()
            _, _, contrastive_loss2 = process_and_project(feature_u[-1], feature_k[-1], proj_mlp2,False)
            contrastive_loss2 = contrastive_loss2.mean()
            contrastive_loss=(contrastive_loss1+contrastive_loss2)/2


            loss_u = l_l2u_u + l_u2l_u + 0.001*contrastive_loss + 0.1*l_fp_u
            loss_k = l_l2u_k + l_u2l_k + 0.001*contrastive_loss + 0.1*l_fp_k

            loss = loss_u + loss_k

            optimizer_u.zero_grad()
            optimizer_k.zero_grad()
            optimizer_mlp1.zero_grad()
            optimizer_mlp2.zero_grad()

            loss.backward()

            optimizer_u.step()
            optimizer_k.step()
            optimizer_mlp1.step()
            optimizer_mlp2.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer_k.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            writer.add_scalar('info/UNet_total_loss', loss_u, iter_num)
            writer.add_scalar('info/KNet_total_loss', loss_k, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)

            logging.info('U_Net iteration %d: loss: %f, l_l2u: %f, l_u2l: %f,l_fp_u: %f' % (
                iter_num, loss_u, l_l2u_u, l_u2l_u,l_fp_u))
            logging.info('K_Net iteration %d: loss: %f, l_l2u: %f, l_u2l: %f,l_fp_k: %f' % (
                iter_num, loss_k, l_l2u_k, l_u2l_k,l_fp_k))
            logging.info('iteration %d: total loss: %f,contra loss:%f' % (iter_num, loss,contrastive_loss))

            if iter_num > 0 and iter_num % 200 == 0:
                model_u.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model_u,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance_u = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance_u, iter_num)

                if performance_u > best_performance_u:
                    best_performance_u = performance_u
                    save_mode_path = os.path.join(snapshot_path_u,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance_u, 4)))
                    save_best_path = os.path.join(snapshot_path_u, '{}_best_model.pth'.format(args.model))
                    torch.save(model_u.state_dict(), save_mode_path)
                    torch.save(model_u.state_dict(), save_best_path)

                logging.info('iteration %d : U-Net mean_dice : %f' % (iter_num, performance_u))

                model_k.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model_k,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance_k = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance_k, iter_num)

                if performance_k > best_performance_k:
                    best_performance_k = performance_k
                    save_mode_path = os.path.join(snapshot_path_k,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance_k, 4)))
                    save_best_path = os.path.join(snapshot_path_k, '{}_best_model.pth'.format(args.model))
                    torch.save(model_k.state_dict(), save_mode_path)
                    torch.save(model_k.state_dict(), save_best_path)

                logging.info('iteration %d : K-Net mean_dice : %f' % (iter_num, performance_k))

                model_u.train()
                model_k.train()

            if iter_num >= max_iterations:
                break

        scheduler_mlp1.step()
        scheduler_mlp2.step()

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path_u = "./model/MCMCCL/ACDC_{}_{}_labeled/pre_train_u".format(args.exp, args.labelnum)
    pre_snapshot_path_k = "./model/MCMCCL/ACDC_{}_{}_labeled/pre_train_k".format(args.exp, args.labelnum)
    self_snapshot_path_u = "./model/MCMCCL/ACDC_{}_{}_labeled/self_train_u".format(args.exp, args.labelnum)
    self_snapshot_path_k = "./model/MCMCCL/ACDC_{}_{}_labeled/self_train_k".format(args.exp, args.labelnum)

    for snapshot_path in [pre_snapshot_path_u, pre_snapshot_path_k, self_snapshot_path_u, self_snapshot_path_k]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('../code/ACDC_train.py', self_snapshot_path_u)
    shutil.copy('../code/ACDC_train.py', pre_snapshot_path_k)


    # Pre_train K-Net
    logging.basicConfig(filename=pre_snapshot_path_k + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train_k(args, pre_snapshot_path_k)


    # Pre_train U-Net
    logging.basicConfig(filename=pre_snapshot_path_u + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train_u(args, pre_snapshot_path_u)


    # Self_train
    logging.basicConfig(filename=self_snapshot_path_u + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path_u, pre_snapshot_path_k, self_snapshot_path_u, self_snapshot_path_k)





