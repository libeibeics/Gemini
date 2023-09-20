import os
import time
import argparse
import torch
from torch import nn
from torch.backends import cudnn
import dataset as dataset
import numpy as np
import torchvision
import torch.nn.functional as F
from wideresnet_feat import WideResNet
from partial_models.resnet import resnet
from resnet_feat import resnet18, resnet34, resnet50, resnet101
from lenet import LeNet
import logging
import copy

parser = argparse.ArgumentParser(description='Revisiting Consistency Regularization for Deep Partial Label Learning')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--n_warmup_epoch', default=20, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--lam', default=1, type=float)

parser.add_argument('--dataset', type=str, choices=['svhn', 'cifar10', 'cifar100', 'fmnist', 'kmnist'],
                    default='svhn')

parser.add_argument('--model', type=str, choices=['resnet', 'resnet18', 'widenet', 'lenet'], default='resnet18')

parser.add_argument('--lr', default=0.1, type=float)

parser.add_argument('--rate', default=0.1, type=float, help='-1 for feature, 0.x for random')

parser.add_argument('--trial', default='1', type=str)

parser.add_argument('--data-dir', default='./data/', type=str)

parser.add_argument('--note', default='', type=str)

parser.add_argument('--lam_type', type=str, choices=['fixed', 'adaptive'], default='adaptive')

parser.add_argument('--tau', default=20, type=float, help='temperature parameter')

parser.add_argument('--log_path', type=str, default='log')

parser.add_argument('--sim_tau', default=0.1, type=float, help='tau when calculate similarity')
parser.add_argument('--w_conf_struct', default=1, type=float, help='weight of confidence-based structural loss')
parser.add_argument('--w_sim_struct', default=1, type=float, help='weight of similarity-based structural loss')

args = parser.parse_args()

num_classes = 100 if args.dataset == 'cifar100' else 10

log_path = '{}{}'.format(args.log_path, args.trial)

if not os.path.exists(log_path): os.makedirs(log_path)
curr_filename = os.path.basename(__file__).split('.')[0]
args.filename = curr_filename + '_{}_{}_p{}_tau{}_{}_{}'.format(args.model, args.dataset, args.rate, args.tau,
                                                                args.note,
                                                                time.strftime("%m%d%H%M%S",
                                                                              time.localtime()))

if 'sim_tau' in args.log_path:
    args.filename = curr_filename + '_{}_{}_p{}_sim_tau{}_{}_{}'.format(args.model, args.dataset, args.rate,
                                                                                 args.sim_tau,
                                                                                 args.note, time.strftime("%m%d%H%M%S",
                                                                                                          time.localtime()))

logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    filename="{}/{}.log".format(log_path, args.filename),
                    level=logging.INFO)

logging.info(args)



'''
添加 structural similarity
'''
start_point = {
    'cifar10_0.1': 0.97,
    'cifar10_0.3': 0.95,
    'cifar10_0.5': 0.90,
    'cifar10_0.7': 0.80,
    'cifar10_-1.0': 0.95,
    'svhn_0.1': 0.98,
    'svhn_0.3': 0.97,
    'svhn_0.5': 0.96,
    'svhn_0.7': 0.95,
    'svhn_-1.0': 0.97,
    'cifar100_0.01': 0.97,
    'cifar100_0.05': 0.92,
    'cifar100_0.1': 0.83,
    'cifar100_0.2': 0.60,
    'cifar100_-1.0': 0.95
}

if num_classes == 10:
    rate_schedule = list(np.linspace(start_point['{}_{}'.format(args.dataset, args.rate)], 0.99, 50)) + [1] * 150
elif num_classes == 100:
    if args.rate == 0.2:
        args.n_warmup_epoch = 50
        rate_schedule = list(np.linspace(start_point['{}_{}'.format(args.dataset, args.rate)], 0.9, 100)) + [0.9] * 100
    elif args.rate == 0.1:
        rate_schedule = list(np.linspace(start_point['{}_{}'.format(args.dataset, args.rate)], 0.95, 100)) + [
            0.95] * 100
    else:
        rate_schedule = list(np.linspace(start_point['{}_{}'.format(args.dataset, args.rate)], 1, 100)) + [1] * 100


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def rc_loss(logsm_outputs, confidence, reduce=True):
    final_outputs = logsm_outputs * confidence
    average_loss = - ((final_outputs).sum(dim=1))
    if reduce:
        average_loss = average_loss.mean()
    return average_loss


def get_train_accuracy(confidence, labels):
    return torch.mean((torch.max(confidence, dim=-1)[1] == labels).float())


def loss_structure_kl(feat1, feat2):
    # feat1: guide
    # feat2: guided
    feat1 = F.normalize(feat1, dim=-1, p=2)
    feat2 = F.normalize(feat2, dim=-1, p=2)
    feat1_sim = torch.matmul(feat1, feat1.transpose(0, 1)) / args.sim_tau
    feat2_sim = torch.matmul(feat2, feat2.transpose(0, 1)) / args.sim_tau
    feat1_prob = torch.softmax(feat1_sim, dim=-1)
    feat2_log_prob = torch.log_softmax(feat2_sim, dim=-1)
    return F.kl_div(feat2_log_prob, feat1_prob.detach(), reduction="batchmean")


def loss_structure_binary(feat, sim_matrix):
    feat = F.normalize(feat, dim=-1, p=2)
    feat_sim = torch.sigmoid(torch.matmul(feat, feat.transpose(0, 1)) / 0.1)
    return F.binary_cross_entropy(feat_sim.reshape(-1), sim_matrix.reshape(-1).float())


# def get_train_accuracy(pred_aug0_a, pred_aug1_a, pred_aug2_a, pred_aug0_b, pred_aug1_b, pred_aug2_b, labels):
#     # fix bug
#
#     indicator0 = (torch.max(pred_aug0_a, dim=-1)[1] == labels).float() + (
#             torch.max(pred_aug0_b, dim=-1)[1] == labels).float()
#     indicator1 = (torch.max(pred_aug1_a, dim=-1)[1] == labels).float() + (
#             torch.max(pred_aug1_b, dim=-1)[1] == labels).float()
#     indicator2 = (torch.max(pred_aug2_a, dim=-1)[1] == labels).float() + (
#             torch.max(pred_aug2_b, dim=-1)[1] == labels).float()
#     indicators = torch.cat([indicator0, indicator1, indicator2], dim=0)
#     return torch.mean(indicators) / 2


def pick_smallloss(nll_losses, rc_losses, align_losses, co_lambda, remember_rate, noise_or_not):
    loss_pick = nll_losses * (1 - co_lambda) + co_lambda * align_losses
    ind_sorted = torch.argsort(loss_pick)
    loss_sorted = loss_pick[ind_sorted]
    num_remember = int(remember_rate * loss_sorted.shape[0])
    pure_ratio = torch.mean(noise_or_not[ind_sorted[:num_remember]].float())
    ind_update = ind_sorted[:num_remember]

    # 保留部分
    losses = nll_losses + rc_losses + align_losses
    picked_losses = torch.mean(losses[ind_update])
    picked_nll_losses = torch.mean(nll_losses[ind_update])
    picked_rc_losses = torch.mean(rc_losses[ind_update])
    picked_align_losses = torch.mean(align_losses[ind_update])
    return picked_losses, pure_ratio, picked_nll_losses, picked_rc_losses, picked_align_losses


def DPLL_coteaching_train(train_loader, model_a, model_b, optimizer, epoch, consistency_criterion, confidence):
    """
        Run one train epoch
    """
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    suplosses = AverageMeter()
    conslosses = AverageMeter()
    alignlosses = AverageMeter()
    correct_ratios = AverageMeter()
    picked_pure_ratios = AverageMeter()
    picked_nll_losses = AverageMeter()
    picked_rc_losses = AverageMeter()
    picked_align_losses = AverageMeter()
    conf_struct_losses = AverageMeter()
    sim_struct_losses = AverageMeter()
    same_ratios = AverageMeter()
    end = time.time()

    model_a.train()
    model_b.train()

    for i, (x_aug0, x_aug1, x_aug2, y, part_y, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # partial label
        part_y = part_y.float().cuda()
        # label
        y = y.cuda()
        # original samples with pre-processing
        x_aug0 = x_aug0.cuda()
        x_aug0_feat_a, y_pred_aug0_a = model_a(x_aug0)
        x_aug0_feat_b, y_pred_aug0_b = model_b(x_aug0)

        # augmentation1
        x_aug1 = x_aug1.cuda()
        x_aug1_feat_a, y_pred_aug1_a = model_a(x_aug1)
        x_aug1_feat_b, y_pred_aug1_b = model_b(x_aug1)

        # augmentation2
        x_aug2 = x_aug2.cuda()
        x_aug2_feat_a, y_pred_aug2_a = model_a(x_aug2)
        x_aug2_feat_b, y_pred_aug2_b = model_b(x_aug2)

        y_pred_aug0_a_probas_log = torch.log_softmax(y_pred_aug0_a / args.tau, dim=-1)
        y_pred_aug1_a_probas_log = torch.log_softmax(y_pred_aug1_a / args.tau, dim=-1)
        y_pred_aug2_a_probas_log = torch.log_softmax(y_pred_aug2_a / args.tau, dim=-1)

        y_pred_aug0_a_probas = torch.softmax(y_pred_aug0_a / args.tau, dim=-1)
        y_pred_aug1_a_probas = torch.softmax(y_pred_aug1_a / args.tau, dim=-1)
        y_pred_aug2_a_probas = torch.softmax(y_pred_aug2_a / args.tau, dim=-1)

        y_pred_aug0_b_probas_log = torch.log_softmax(y_pred_aug0_b / args.tau, dim=-1)
        y_pred_aug1_b_probas_log = torch.log_softmax(y_pred_aug1_b / args.tau, dim=-1)
        y_pred_aug2_b_probas_log = torch.log_softmax(y_pred_aug2_b / args.tau, dim=-1)

        y_pred_aug0_b_probas = torch.softmax(y_pred_aug0_b / args.tau, dim=-1)
        y_pred_aug1_b_probas = torch.softmax(y_pred_aug1_b / args.tau, dim=-1)
        y_pred_aug2_b_probas = torch.softmax(y_pred_aug2_b / args.tau, dim=-1)

        '''for comparison'''
        y_pred_probas_a = torch.cat([y_pred_aug0_a_probas, y_pred_aug1_a_probas, y_pred_aug2_a_probas], dim=0)
        y_pred_probas_b = torch.cat([y_pred_aug0_b_probas, y_pred_aug1_b_probas, y_pred_aug2_b_probas], dim=0)
        pred_label_a = torch.max(y_pred_probas_a, dim=-1)[1]
        pred_label_b = torch.max(y_pred_probas_b, dim=-1)[1]
        same_ratios.update(torch.mean((pred_label_a == pred_label_b).float()))

        base_confidence_a = torch.tensor(confidence[index]).float().cuda()
        consist_loss0_a = consistency_criterion(y_pred_aug0_a_probas_log, base_confidence_a, reduce=False)
        consist_loss1_a = consistency_criterion(y_pred_aug1_a_probas_log, base_confidence_a, reduce=False)
        consist_loss2_a = consistency_criterion(y_pred_aug2_a_probas_log, base_confidence_a, reduce=False)

        base_confidence_b = part_y.clone() * y_pred_aug0_b_probas.clone().detach()
        base_confidence_b = base_confidence_b / torch.sum(base_confidence_b + 0.0000001, dim=1, keepdim=True)
        selftrain_loss0_b = consistency_criterion(y_pred_aug0_b_probas_log, base_confidence_b)
        selftrain_loss1_b = consistency_criterion(y_pred_aug1_b_probas_log, base_confidence_b)
        selftrain_loss2_b = consistency_criterion(y_pred_aug2_b_probas_log, base_confidence_b)

        consist_loss = consist_loss0_a + consist_loss1_a + consist_loss2_a
        selftrain_loss_b = selftrain_loss0_b + selftrain_loss1_b + selftrain_loss2_b

        # supervised loss
        # 考虑三个样本所形成的 super loss
        super_loss_a = - torch.log(
            (F.softmax(y_pred_aug0_a / args.tau, dim=1) * part_y).sum(dim=1) + 0.000001).mean() - torch.log(
            (F.softmax(y_pred_aug1_a / args.tau, dim=1) * part_y).sum(dim=1) + 0.000001).mean() - torch.log(
            (F.softmax(y_pred_aug2_a / args.tau, dim=1) * part_y).sum(dim=1) + 0.000001).mean()

        # 按照 clean data 来训练
        pseudo_label = torch.argmax(base_confidence_a, dim=-1)
        super_loss_b = F.cross_entropy(y_pred_aug0_b / args.tau, pseudo_label, reduction='none') + F.cross_entropy(
            y_pred_aug1_b / args.tau, pseudo_label, reduction='none') + F.cross_entropy(y_pred_aug2_b / args.tau,
                                                                                        pseudo_label, reduction='none')
        noise_or_not = pseudo_label == y
        alignlosses = (F.kl_div(y_pred_aug0_a_probas_log, y_pred_aug0_b_probas, reduction='none')
                       + F.kl_div(y_pred_aug0_b_probas_log, y_pred_aug0_a_probas, reduction='none')) / 2

        alignlosses = torch.sum(alignlosses, dim=-1)

        picked_losses, pure_ratio, picked_nll_loss, picked_rc_loss, picked_align_loss = \
            pick_smallloss(super_loss_b, consist_loss, alignlosses, 0.1, rate_schedule[epoch], noise_or_not)

        # confidence-based structural similarity
        conf_struct_loss = (loss_structure_kl(base_confidence_a, x_aug0_feat_a) +
                            loss_structure_kl(base_confidence_a, x_aug1_feat_a) +
                            loss_structure_kl(base_confidence_a, x_aug2_feat_a)) / 3

        lam = min((epoch / 100) * args.lam, args.lam * 0.5)

        # label similarity
        pseudo_onehot_label = torch.zeros_like(base_confidence_a)
        pseudo_onehot_label[torch.arange(x_aug0.shape[0]), pseudo_label] = 1
        similarity_matrix = pseudo_onehot_label.matmul(pseudo_onehot_label.transpose(0, 1))
        sim_struct_loss = (loss_structure_binary(x_aug0_feat_b, similarity_matrix) \
                           + loss_structure_binary(x_aug1_feat_b, similarity_matrix) \
                           + loss_structure_binary(x_aug2_feat_b, similarity_matrix)) / 3

        # 添加两个模型之间的互相蒸馏
        '''
        1. 噪声标记中损失越大，说明越可能是噪声，则PLL的损失反传梯度时权重应越小
        2. 噪声标记一支中只对损失较小的数据进行反传梯度，这样模型近似于干净数据训练得到，然后用该模型的计算结果对PLL模型的计算结果进行蒸馏。
        
        不受影响的损失
        1. CC 损失
        2. 自监督损失
        
        受影响的损失
        1. RC 损失
        2. 噪声标记中的交叉熵损失
        3. 两个模型之间的蒸馏损失
        
        取最小损失：1. 噪声标记交叉熵损失，2.（可选）两者对齐的损失
        '''

        final_loss = super_loss_a + picked_losses + selftrain_loss_b + lam * args.w_conf_struct * conf_struct_loss \
                     + lam * 0.5 * args.w_sim_struct * sim_struct_loss
        correct_ratio = get_train_accuracy(base_confidence_a, y)
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # update confidence
        confidence_update(confidence, y_pred_aug0_a_probas, y_pred_aug1_a_probas, y_pred_aug2_a_probas, part_y, index)
        conf_struct_losses.update(conf_struct_loss.item(), x_aug0.size(0))
        sim_struct_losses.update(sim_struct_loss.item(), x_aug0.size(0))
        picked_pure_ratios.update(pure_ratio.item(), int(rate_schedule[epoch] * x_aug0.size(0)))
        picked_nll_losses.update(picked_nll_loss.item(), int(rate_schedule[epoch] * x_aug0.size(0)))
        picked_rc_losses.update(picked_rc_loss.item(), int(rate_schedule[epoch] * x_aug0.size(0)))
        picked_align_losses.update(picked_align_loss.item(), int(rate_schedule[epoch] * x_aug0.size(0)))
        correct_ratios.update(correct_ratio.item(), x_aug0.size(0))
        losses.update(final_loss.item(), x_aug0.size(0))
        suplosses.update(super_loss_a.item(), x_aug0.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 200 == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'CorrectRatio {correct_ratios.val:.4f} ({correct_ratios.avg:.4f})\t'
                         'PickedPureRatio {pure_ratios.val:.4f} ({pure_ratios.avg:.4f})\t'
                         'SameRatio {same_ratios.val:.4f} ({same_ratios.avg:.4f})\t'
                         'ConfStructLosses {conf_struct_losses.val:.4f} ({conf_struct_losses.avg:.4f})\t'
                         'SimStructLosses {sim_struct_losses.val:.4f} ({sim_struct_losses.avg:.4f})\t'
                         'PickedNLLLosses {picked_nll_losses.val:.4f} ({picked_nll_losses.avg:.4f})\t'
                         'PickedRCLosses {picked_rc_losses.val:.4f} ({picked_rc_losses.avg:.4f})\t'
                         'PickedAlignLosses {picked_align_losses.val:.4f} ({picked_align_losses.avg:.4f})\t'
                         'SameRatio {same_ratios.val:.4f} ({same_ratios.avg:.4f})\t'
                         'Lambda ({lam:.4f})\t'
                         'RememberRate ({remember_rate:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, correct_ratios=correct_ratios, pure_ratios=picked_pure_ratios,
                picked_nll_losses=picked_nll_losses, picked_rc_losses=picked_rc_losses,
                picked_align_losses=picked_align_losses, sim_struct_losses=sim_struct_losses,
                conf_struct_losses=conf_struct_losses,
                same_ratios=same_ratios, lam=lam, remember_rate=rate_schedule[epoch]))
    return losses.avg


def confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y, index):
    y_pred_aug0_probas = y_pred_aug0_probas.detach()
    y_pred_aug1_probas = y_pred_aug1_probas.detach()
    y_pred_aug2_probas = y_pred_aug2_probas.detach()

    revisedY0 = part_y.clone()

    revisedY0 = revisedY0 * torch.pow(y_pred_aug0_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug1_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug2_probas, 1 / (2 + 1))
    revisedY0 = revisedY0 / (revisedY0.sum(dim=1).repeat(num_classes, 1).transpose(0, 1) + 0.0000001)

    confidence[index, :] = revisedY0.cpu().numpy()


def validate_coteaching(valid_loader, model_a, model_b, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_a = AverageMeter()
    top1_b = AverageMeter()
    same_ratios = AverageMeter()
    end = time.time()

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            _, output_a = model_a(input_var)
            loss_a = criterion(output_a, target_var)
            output_a, loss_a = output_a.float(), loss_a.float()

            _, output_b = model_b(input_var)
            loss_b = criterion(output_b, target_var)
            output_b, loss_b = output_b.float(), loss_b.float()

            same_ratios.update(torch.mean((torch.max(output_a, dim=-1)[1] == torch.max(output_b, dim=-1)[1]).float()))

            # measure accuracy and record loss
            prec1 = accuracy((output_a.data + output_b.data) / 2, target)[0]
            prec1_a = accuracy(output_a.data, target)[0]
            prec1_b = accuracy(output_b.data, target)[0]
            losses.update((loss_a.item() + loss_b.item()) / 2, input.size(0))
            top1.update(prec1.item(), input.size(0))
            top1_a.update(prec1_a.item(), input.size(0))
            top1_b.update(prec1_b.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    logging.info('Test: [{0}/{1}]\t'
                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                 'PrecA@1 {top1_a.val:.3f} ({top1_a.avg:.3f})\t'
                 'PrecB@1 {top1_b.val:.3f} ({top1_b.avg:.3f})'.format(
        i, len(valid_loader), batch_time=batch_time, loss=losses, top1=top1, top1_a=top1_a, top1_b=top1_b))

    logging.info(
        ' * Prec@1 {top1.avg:.3f}, PrecA@1 {top1_a.avg:.3f}, PrecB@1 {top1_b.avg:.3f}, SameRatio:{samerate.avg:.3f}'.format(
            top1=top1, top1_a=top1_a, top1_b=top1_b, samerate=same_ratios))
    return top1.avg, losses.avg


def validate(valid_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            _, output = model(input_var)
            loss = criterion(output, target_var)
            output, loss = output.float(), loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    logging.info('Test: [{0}/{1}]\t'
                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        i, len(valid_loader), batch_time=batch_time, loss=losses, top1=top1))

    logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg


def Warmup_train(train_loader, model, optimizer, epoch, consistency_criterion, confidence):
    """
        Run one train epoch
    """
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    suplosses = AverageMeter()
    conslosses = AverageMeter()
    correct_ratios = AverageMeter()
    same_ratios = AverageMeter()
    end = time.time()

    model.train()

    for i, (x_aug0, x_aug1, x_aug2, y, part_y, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # partial label
        part_y = part_y.float().cuda()
        # label
        y = y.cuda()
        # original samples with pre-processing
        x_aug0 = x_aug0.cuda()
        _, y_pred_aug0 = model(x_aug0)

        # augmentation1
        x_aug1 = x_aug1.cuda()
        _, y_pred_aug1 = model(x_aug1)

        # augmentation2
        x_aug2 = x_aug2.cuda()
        _, y_pred_aug2 = model(x_aug2)

        y_pred_aug0_probas_log = torch.log_softmax(y_pred_aug0 / args.tau, dim=-1)
        y_pred_aug1_probas_log = torch.log_softmax(y_pred_aug1 / args.tau, dim=-1)
        y_pred_aug2_probas_log = torch.log_softmax(y_pred_aug2 / args.tau, dim=-1)

        y_pred_aug0_probas = torch.softmax(y_pred_aug0 / args.tau, dim=-1)
        y_pred_aug1_probas = torch.softmax(y_pred_aug1 / args.tau, dim=-1)
        y_pred_aug2_probas = torch.softmax(y_pred_aug2 / args.tau, dim=-1)

        base_confidence = torch.tensor(confidence[index]).float().cuda()
        consist_loss0_a = consistency_criterion(y_pred_aug0_probas_log, base_confidence)
        consist_loss1_a = consistency_criterion(y_pred_aug1_probas_log, base_confidence)
        consist_loss2_a = consistency_criterion(y_pred_aug2_probas_log, base_confidence)

        consist_loss = consist_loss0_a + consist_loss1_a + consist_loss2_a

        correct_ratio = get_train_accuracy(base_confidence, y)
        # supervised loss
        # 考虑三个样本所形成的 super loss
        super_loss = - torch.log(
            (F.softmax(y_pred_aug0 / args.tau, dim=1) * part_y).sum(dim=1) + 0.000001).mean() - torch.log(
            (F.softmax(y_pred_aug1 / args.tau, dim=1) * part_y).sum(dim=1) + 0.000001).mean() - torch.log(
            (F.softmax(y_pred_aug2 / args.tau, dim=1) * part_y).sum(dim=1) + 0.000001).mean()

        # dynamic lam
        lam = min((epoch / 100) * args.lam, args.lam)

        if args.lam_type == 'fixed':
            lam = args.lam

        final_loss = super_loss + lam * consist_loss

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # update confidence
        confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y, index)

        correct_ratios.update(correct_ratio.item(), x_aug0.size(0))
        losses.update(final_loss.item(), x_aug0.size(0))
        suplosses.update(super_loss.item(), x_aug0.size(0))
        conslosses.update(consist_loss.item(), x_aug0.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 200 == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'CorrectRatio {correct_ratios.val:.4f} ({correct_ratios.avg:.4f})\t'
                         'lam ({lam})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, correct_ratios=correct_ratios, lam=lam))
    return losses.avg


def DPLL_Coteaching():
    global args
    # load data
    if args.dataset == "cifar10":
        train_loader, test = dataset.cifar10_dataloaders(args.data_dir, args.rate)
        channel = 3
    elif args.dataset == 'svhn':
        train_loader, test = dataset.svhn_dataloaders(args.data_dir + "svhn/", args.rate)
        channel = 3
    elif args.dataset == 'cifar100':
        train_loader, test = dataset.cifar100_dataloaders(args.data_dir, args.rate)
        channel = 3
    elif args.dataset == 'fmnist':
        train_loader, test = dataset.fmnist_dataloaders(args.data_dir, args.rate)
        channel = 1
    elif args.dataset == 'kmnist':
        train_loader, test = dataset.kmnist_dataloaders(args.data_dir, args.rate)
        channel = 1
    else:
        assert "Unknown dataset"

    # load model
    if args.model == 'widenet':
        model_a = WideResNet(34, num_classes, widen_factor=10, dropRate=0.0)
        model_b = WideResNet(34, num_classes, widen_factor=10, dropRate=0.0)
    elif args.model == 'resnet':
        model_a = resnet(depth=32, n_outputs=num_classes)
        model_b = resnet(depth=32, n_outputs=num_classes)
    elif args.model == 'resnet18':
        model_a = resnet18(n_outputs=num_classes)
        model_b = resnet18(n_outputs=num_classes)
    elif args.model == 'lenet':
        model_a = LeNet(out_dim=num_classes, in_channel=1, img_sz=28)
        model_b = LeNet(out_dim=num_classes, in_channel=1, img_sz=28)
    else:
        assert "Unknown model"
    model_a = model_a.cuda()
    model_b = model_b.cuda()

    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    consistency_criterion = None
    consistency_criterion = rc_loss
    # optimizer
    optimizer_warmup = torch.optim.SGD(model_a.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.SGD(list(model_a.parameters()) + list(model_b.parameters()), lr=args.lr, momentum=0.9,
                                weight_decay=1e-4)
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)
    scheduler_warmup = torch.optim.lr_scheduler.MultiStepLR(optimizer_warmup, milestones=[100, 150], last_epoch=-1)

    cudnn.benchmark = True
    # init confidence
    confidence = copy.deepcopy(train_loader.dataset.partial_labels)
    confidence = confidence / confidence.sum(axis=1)[:, None]

    # Train loop
    valacc_list = []

    # warmup
    for epoch in range(args.n_warmup_epoch):
        Warmup_train(train_loader, model_a, optimizer_warmup, epoch, consistency_criterion,
                     confidence)

        valacc, valloss = validate(test, model_a, criterion, epoch)
        scheduler_warmup.step()

    model_b.load_state_dict(model_a.state_dict())

    for epoch in range(0, args.epochs):
        logging.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        # training
        trainloss = DPLL_coteaching_train(train_loader, model_a, model_b, optimizer, epoch, consistency_criterion,
                                          confidence)

        # evaluate on validation set
        valacc, valloss = validate_coteaching(test, model_a, model_b, criterion, epoch)
        scheduler.step()

        if epoch >= args.epochs - 10:
            valacc_list.append(valacc)
    logging.info('Avg acc of last 10 epochs: {:.4f}'.format(np.average(np.array(valacc_list))))


if __name__ == '__main__':
    DPLL_Coteaching()
