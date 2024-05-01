import logging
import os

import numpy as np
from torch import nn
import torch
import random
import torch.nn.functional as F

# this function guarantees reproductivity
# other packages also support seed options, you can add to this function
def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def kaiming_normal_init_weight(model):
    # for 2d
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

class DistillationLoss(nn.Module):
    def __init__(self, temp: float):
        super(DistillationLoss, self).__init__()
        self.T = temp

    def forward(self, out1, out2, ignore=None):
        # loss = F.kl_div(
        #     F.log_softmax(out1 / self.T, dim=1),
        #     F.softmax(out2 / self.T, dim=1),
        #     reduction="none",
        # )
        # use ignore to mask the loss
        # print(ignore.shape)
        
        # if ignore is not None:
        #     out1 = out1 * ignore
        #     out2 = out2 * ignore
            
        # print(out1[ignore.unsqueeze(0).repeat(out1.shape[0], 1, 1, 1) != 1].shape)
        # if ignore is None:
        #     ignore = 1
        loss = F.kl_div(
            F.log_softmax(out1 / self.T, dim=1),
            F.softmax(out2 / self.T, dim=1),
            reduction="none",
        )
        
        # assert loss  
        assert torch.isnan(loss).sum() == 0, print(loss)
        return loss


class MSELoss(nn.Module):
    def __init__(self, reduction):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.mseloss = nn.MSELoss(reduction=reduction)

    def forward(self, out1, out2, ignore=None):
        
        if ignore is not None:
            out1 = out1 * ignore
            out2 = out2 * ignore
            
        loss = self.mseloss(out1, out2)
        assert torch.isnan(loss).sum() == 0, print(loss)
        return loss


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
        smooth = 1e-4
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

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
