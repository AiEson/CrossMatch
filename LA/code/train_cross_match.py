import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from utils.losses import dice_loss
from dataloaders.la_heart import (
    LAHeart,
    RandomCrop,
    RandomRotFlip,
    ToTensor
)
from test_util import test_all_case
from utils import ramps
import time



parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="../data/2018LA_Seg_Training Set/",
    help="Name of Experiment",
)
parser.add_argument("--exp", type=str, default="cross_match", help="model_name")
parser.add_argument(
    "--max_iterations", type=int, default=9000, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=2, help="batch_size per gpu")
parser.add_argument(
    "--base_lr", type=float, default=0.01, help="maximum epoch number to train"
)
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--label_num", type=int, default=16, help="label num")
parser.add_argument(
    "--eta", type=float, default=0.3, help="weight to balance loss"
)
parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer")
parser.add_argument("--conf_thresh", type=float, default=0.75, help="conf_thresh")
parser.add_argument("--temperature", type=float, default=1, help="temperature")

parser.add_argument('--pert_gap', type=float, default=0.5, help='the perturbation gap')
parser.add_argument('--pert_type', type=str, default='dropout', help='feature pertubation types')


args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(","))
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
else:
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

patch_size = (112, 112, 80)
num_classes = 2

LABELED_ID_NUM = args.label_num  # 8 or 16
conf_thresh = args.conf_thresh
eta = args.eta
pert_gap = args.pert_gap

pervious_bset_dice = 0.0

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + "/code"):
        shutil.rmtree(snapshot_path + "/code")

    shutil.copytree(
        ".", snapshot_path + "/code", shutil.ignore_patterns([".git", "__pycache__"])
    )

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = VNet(
        n_channels=1, n_classes=num_classes, normalization="batchnorm", has_dropout=True, pert_gap=pert_gap,
        pert_type=args.pert_type
    )
    net = net.cuda()
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainset_u = LAHeart(
        base_dir=train_data_path,
        mode="train_u",
        num=80 - LABELED_ID_NUM,
        transform=transforms.Compose(
            [
                RandomRotFlip(),
                RandomCrop(patch_size),
                ToTensor(),
            ]
        ),
        id_path=f"../data/2018LA_Seg_Training Set/train_{LABELED_ID_NUM}_unlabel.list",
    )
    trainsampler_u = torch.utils.data.sampler.RandomSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size * 2,
        drop_last=True,
        sampler=trainsampler_u,
        worker_init_fn=worker_init_fn
    )
    trainsampler_u_mix = torch.utils.data.sampler.RandomSampler(trainset_u, replacement=True)
    trainloader_u_mix = DataLoader(
        trainset_u,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size * 2,
        drop_last=True,
        sampler=trainsampler_u_mix,
        worker_init_fn=worker_init_fn
    )

    trainset_l = LAHeart(
        base_dir=train_data_path,
        mode="train_l",
        num=80 - LABELED_ID_NUM,
        transform=transforms.Compose(
            [
                RandomRotFlip(),
                RandomCrop(patch_size),
                ToTensor(),
            ]
        ),
        id_path=f"../data/2018LA_Seg_Training Set/train_{LABELED_ID_NUM}_label.list",
    )
    trainsampler_l = torch.utils.data.sampler.RandomSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size * 2,
        drop_last=True,
        sampler=trainsampler_l,
        worker_init_fn=worker_init_fn
    )

    net.train()
    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0002 if LABELED_ID_NUM == 16 else 0.0001
        )
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(
            net.parameters(), lr=base_lr, weight_decay=0.0001
        )
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(
            net.parameters(), lr=base_lr, weight_decay=0.0001
        )
    else:
        raise NotImplementedError
    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} itertations per epoch".format(len(trainloader_l)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader_l) + 1
    print(f"All Epochs: {max_epoch}")
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):

        time1 = time.time()
        net.train()

        for i_batch, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
            (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _),
        ) in enumerate(zip(trainloader_l, trainloader_u, trainloader_u_mix)):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            
            with torch.no_grad():
                net.eval()
                pred_u_w_mix = net(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
                
                _, _, pred_u_w_weak_mix, pred_u_w_strong_mix = net(
                    torch.cat((img_x, img_u_w_mix)),
                    need_fp=True
                )
                conf_u_w_weak_mix = pred_u_w_weak_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_weak_mix = pred_u_w_weak_mix.argmax(dim=1)
                
                conf_u_w_strong_mix = pred_u_w_strong_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_strong_mix = pred_u_w_strong_mix.argmax(dim=1)

            img_u_s1[
                cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1
            ] = img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[
                cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1
            ] = img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            net.train()

            pred_x, pred_u_w, pred_u_w_weak, pred_u_w_strong = net(
                torch.cat((img_x, img_u_w)),
                need_fp=True
            )
            if args.temperature != 1:
                pred_u_w = pred_u_w / args.temperature
                pred_u_w_weak = pred_u_w_weak / args.temperature
                pred_u_w_strong = pred_u_w_strong / args.temperature


            pred_u_s1, pred_u_s2, pred_u_s2_weak, pred_u_s2_strong = net(torch.cat((img_u_s1, img_u_s2)), need_fp=True)
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)
            
            # pred_u_w_weak = pred_u_w_weak.detach()
            conf_u_w_weak = pred_u_w_weak.softmax(dim=1).max(dim=1)[0]
            mask_u_w_weak = pred_u_w_weak.argmax(dim=1)
            
            # pred_u_w_strong = pred_u_w_strong.detach()
            conf_u_w_strong = pred_u_w_strong.softmax(dim=1).max(dim=1)[0]
            mask_u_w_strong = pred_u_w_strong.argmax(dim=1)
            
            # pred_u_s2_weak = pred_u_s2_weak.detach()
            conf_u_s2_weak = pred_u_s2_weak.softmax(dim=1).max(dim=1)[0]
            mask_u_s2_weak = pred_u_s2_weak.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()
            mask_u_w_weak_cutmixed2, conf_u_w_weak_cutmixed2 = mask_u_w_weak.clone(), conf_u_w_weak.clone()
            mask_u_w_strong_cutmixed2, conf_u_w_strong_cutmixed2 = mask_u_w_strong.clone(), conf_u_w_strong.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            
            mask_u_w_weak_cutmixed2[cutmix_box2 == 1] = mask_u_w_weak_mix[cutmix_box2 == 1]
            conf_u_w_weak_cutmixed2[cutmix_box2 == 1] = conf_u_w_weak_mix[cutmix_box2 == 1]
            
            mask_u_w_strong_cutmixed2[cutmix_box2 == 1] = mask_u_w_strong_mix[cutmix_box2 == 1]
            conf_u_w_strong_cutmixed2[cutmix_box2 == 1] = conf_u_w_strong_mix[cutmix_box2 == 1]
            

            loss_x = (
                F.cross_entropy(pred_x, mask_x)
                + dice_loss(pred_x.softmax(dim=1)[:, 1, :, :, :], mask_x == 1)
            ) / 2.0

            loss_u_s1 = dice_loss(
                pred_u_s1.softmax(dim=1)[:, 1, :, :, :],
                mask_u_w_cutmixed1 == 1,
                ignore=(conf_u_w_cutmixed1 < conf_thresh).float(),
            )

            loss_u_s2 = dice_loss(
                pred_u_s2.softmax(dim=1)[:, 1, :, :, :],
                mask_u_w_cutmixed2 == 1,
                ignore=(conf_u_w_cutmixed2 < conf_thresh).float(),
            )
            
            # pred_u_w_weak to supervise pred_u_s2_weak
            loss_weak_dec_w_s2 = dice_loss(
                pred_u_s2_weak.softmax(dim=1)[:, 1, :, :, :],
                mask_u_w_weak_cutmixed2 == 1,
                ignore=(conf_u_w_weak_cutmixed2 < conf_thresh).float(),
            )
            # pred_u_w_strong to supervise pred_u_s2_strong
            loss_strong_dec_w_s2 = dice_loss(
                pred_u_s2_strong.softmax(dim=1)[:, 1, :, :, :],
                mask_u_w_strong_cutmixed2 == 1,
                ignore=(conf_u_w_strong_cutmixed2 < conf_thresh).float(),
            )
            
            
            ignores = (conf_u_w < conf_thresh).float()
            masks = mask_u_w == 1
            # pred_u_w to supervise pred_u_w_weak
            loss_weak_dec_w_t = dice_loss(
                pred_u_w_weak.softmax(dim=1)[:, 1, :, :, :],
                masks,
                ignore=ignores,
            )
            # pred_u_w to supervise pred_u_w_strong
            loss_strong_dec_w_t = dice_loss(
                pred_u_w_strong.softmax(dim=1)[:, 1, :, :, :],
                masks,
                ignore=ignores,
            )
            # pred_u_w to supervise pred_u_s2_weak
            loss_weak_dec_w_s2_t = dice_loss(
                pred_u_s2_weak.softmax(dim=1)[:, 1, :, :, :],
                mask_u_w_cutmixed2 == 1,
                ignore=(conf_u_w_cutmixed2 < conf_thresh).float(),
            )
            # pred_u_w to supervise pred_u_s2_strong
            loss_strong_dec_w_s2_t = dice_loss(
                pred_u_s2_strong.softmax(dim=1)[:, 1, :, :, :],
                mask_u_w_cutmixed2 == 1,
                ignore=(conf_u_w_cutmixed2 < conf_thresh).float(),
            )
            
            loss_u_w_kd = (1-eta) * (
                loss_weak_dec_w_t + loss_strong_dec_w_t + loss_weak_dec_w_s2_t + loss_strong_dec_w_s2_t
            ) / 4.0 + eta * (loss_weak_dec_w_s2 + loss_strong_dec_w_s2) / 2.0
            
            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_kd) / 2.0
            
            conf_thresh = (
                args.conf_thresh + (1 - args.conf_thresh) * ramps.sigmoid_rampup(iter_num, 17000)
            ) * np.log(2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar("loss/loss_u_kd", loss_u_w_kd, iter_num)
            writer.add_scalar("loss/loss_x", loss_x, iter_num)
            writer.add_scalar("loss/loss", loss, iter_num)
            
            lr_ = base_lr * (1 - iter_num / 17000) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            if iter_num % 50 == 0:
                image = (
                    img_x[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                )
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image("train/Image", grid_image, iter_num)

                outputs_soft = F.softmax(pred_x, 1)
                image = (
                    outputs_soft[0, 1:2, :, :, 20:61:10]
                    .permute(3, 0, 1, 2)
                    .repeat(1, 3, 1, 1)
                )
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image("train/Predicted_label", grid_image, iter_num)

                image = (
                    mask_x[0, :, :, 20:61:10]
                    .unsqueeze(0)
                    .permute(3, 0, 1, 2)
                    .repeat(1, 3, 1, 1)
                )
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image("train/Groundtruth_label", grid_image, iter_num)
                
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, "iter_" + str(iter_num) + ".pth"
                )
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                print("finish training, iter_num > max_iterations")
                break
            time1 = time.time()
        if iter_num > max_iterations:
            print("finish training")
            break

        if (
            epoch_num % 8 == 0
            and 
            # iter_num > max_iterations-1000
            False
            ) or iter_num % 1000 == 0:
            # evals
            net.eval()
            with torch.no_grad():
                with open(args.root_path + "./test.list", "r") as f:
                    image_list = f.readlines()
                image_list = [
                    args.root_path + item.replace("\n", "") + "/mri_norm2.h5"
                    for item in image_list
                ]

                dice, jc, hd, asd = test_all_case(
                    net,
                    image_list,
                    num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=18,
                    stride_z=4,
                    save_result=False,
                    test_save_path=None,
                )

                if dice > pervious_bset_dice:
                    pervious_bset_dice = dice
                    save_mode_path = os.path.join(snapshot_path, "best_model.pth")
                    torch.save(net.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

    save_mode_path = os.path.join(
        snapshot_path, "iter_" + str(max_iterations + 1) + ".pth"
    )
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
