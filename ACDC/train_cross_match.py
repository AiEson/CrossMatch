import argparse
# from line_profiler import LineProfiler
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from util.thresh_helper import ThreshController, DropRateController
from dataset.acdc import ACDCDataset
from model.unet import UNet
from util.soft_dice_loss import dice_loss
from util.classes import CLASSES
from util.utils import (
    AverageMeter,
    count_params,
    init_log,
    DiceLoss,
    seed_everything,
    DistillationLoss,
    MSELoss
)
from util.dist_helper import setup_distributed
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


parser = argparse.ArgumentParser(
    description="Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation"
)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--labeled-id-path", type=str, required=True)
parser.add_argument("--unlabeled-id-path", type=str, required=True)
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--port", default=None, type=int)
parser.add_argument("--eta", default=0.3, type=float)


def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    conf_thresh = cfg["conf_thresh"]
    thresh_controller = ThreshController(
        nclass=4, momentum=0.999 / 4.0, thresh_init=cfg["conf_thresh"]
    )

    drop_rate = cfg["drop_rate"]
    drop_rate_controller = DropRateController(momentum=0.9)

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)
    
    model = UNet(in_chns=1, class_num=cfg["nclass"])
    if rank == 0:
        logger.info("Total params: {:.1f}M\n".format(count_params(model)))

    optimizer = SGD(model.parameters(), cfg["lr"], momentum=0.9, weight_decay=0.0001, nesterov=True)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=False,
    )

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg["nclass"])
    creiterion_kd = DistillationLoss(temp=cfg["T"])
    creiterion_mmd = MSELoss(reduction="none")

    trainset_u = ACDCDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        cfg["crop_size"],
        args.unlabeled_id_path,
    )
    trainset_l = ACDCDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        cfg["crop_size"],
        args.labeled_id_path,
        nsample=len(trainset_u.ids),
    )
    valset = ACDCDataset(cfg["dataset"], cfg["data_root"], "val")

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=cfg["batch_size"]*3,
        drop_last=True,
        sampler=trainsampler_l,
    )
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=cfg["batch_size"]*3,
        drop_last=True,
        sampler=trainsampler_u,
    )
    trainsampler_u_mix = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u_mix = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=cfg["batch_size"]*3,
        drop_last=True,
        sampler=trainsampler_u_mix,
    )
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=3,
        drop_last=False,
        sampler=valsampler,
        pin_memory_device=f"cuda:{local_rank}",
    )

    total_iters = len(trainloader_u) * cfg["epochs"]
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        checkpoint = torch.load(os.path.join(args.save_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]

        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    for epoch in range(epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}, TH: {:.4f}".format(
                    epoch, optimizer.param_groups[0]["lr"], previous_best, conf_thresh
                )
            )

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_kd = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        trainloader_u_mix.sampler.set_epoch(epoch + cfg["epochs"])

        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
            (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _),
        ) in enumerate(loader):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
                
                _, _, pred_u_w_weak_mix, pred_u_w_strong_mix = model(
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

            model.train()

            pred_x, pred_u_w, pred_u_w_weak, pred_u_w_strong = model(
                torch.cat((img_x, img_u_w)),
                need_fp=True
            )
            pred_u_s1, pred_u_s2, pred_u_s2_weak, pred_u_s2_strong = model(torch.cat((img_u_s1, img_u_s2)), need_fp=True)
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)
            
            conf_u_w_weak = pred_u_w_weak.softmax(dim=1).max(dim=1)[0]
            mask_u_w_weak = pred_u_w_weak.argmax(dim=1)
            
            conf_u_w_strong = pred_u_w_strong.softmax(dim=1).max(dim=1)[0]
            mask_u_w_strong = pred_u_w_strong.argmax(dim=1)
            

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
                criterion_ce(pred_x, mask_x)
                + criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())
            ) / 2.0

            loss_u_s1 = criterion_dice(
                (pred_u_s1).softmax(dim=1),
                mask_u_w_cutmixed1.unsqueeze(1).float(),
                ignore=(conf_u_w_cutmixed1 < conf_thresh).float(),
            )

            loss_u_s2 = criterion_dice(
                (pred_u_s2).softmax(dim=1),
                mask_u_w_cutmixed2.unsqueeze(1).float(),
                ignore=(conf_u_w_cutmixed2 < conf_thresh).float(),
            )
            
            # pred_u_w_weak to supervise pred_u_s2_weak
            loss_weak_dec_w_s2 = criterion_dice(
                pred_u_s2_weak.softmax(dim=1),
                mask_u_w_weak_cutmixed2.unsqueeze(1).float(),
                ignore=(conf_u_w_weak_cutmixed2 < conf_thresh).float(),
            )
            # pred_u_w_strong to supervise pred_u_s2_strong
            loss_strong_dec_w_s2 = criterion_dice(
                pred_u_s2_strong.softmax(dim=1),
                mask_u_w_strong_cutmixed2.unsqueeze(1).float(),
                ignore=(conf_u_w_strong_cutmixed2 < conf_thresh).float(),
            )
            
            ignores = (conf_u_w < conf_thresh).float()
            masks = mask_u_w.unsqueeze(1).float()
            # pred_u_w to supervise pred_u_w_weak
            loss_weak_dec_w_t = criterion_dice(
                pred_u_w_weak.softmax(dim=1),
                masks,
                ignore=ignores,
            )
            # pred_u_w to supervise pred_u_w_strong
            loss_strong_dec_w_t = criterion_dice(
                pred_u_w_strong.softmax(dim=1),
                masks,
                ignore=ignores,
            )
            # pred_u_w to supervise pred_u_s2_weak
            loss_weak_dec_w_s2_t = criterion_dice(
                pred_u_s2_weak.softmax(dim=1),
                mask_u_w_cutmixed2.unsqueeze(1).float(),
                ignore=(conf_u_w_cutmixed2 < conf_thresh).float(),
            )
            # pred_u_w to supervise pred_u_s2_strong
            loss_strong_dec_w_s2_t = criterion_dice(
                pred_u_s2_strong.softmax(dim=1),
                mask_u_w_cutmixed2.unsqueeze(1).float(),
                ignore=(conf_u_w_cutmixed2 < conf_thresh).float(),
            )
            
            loss_u_w_kd = (1-args.eta) * (
                loss_weak_dec_w_t + loss_strong_dec_w_t + loss_weak_dec_w_s2_t + loss_strong_dec_w_s2_t
            ) / 4.0 + args.eta * (loss_weak_dec_w_s2 + loss_strong_dec_w_s2) / 2.0
            
            
            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_kd * 0.5) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_kd.update(loss_u_w_kd.item())

            mask_ratio = (conf_u_w >= conf_thresh).sum() / conf_u_w.numel()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i

            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            if cfg["use_threshold_relax"]:
                thresh_controller.thresh_update(
                    pred_u_w.detach().float(),
                    None,
                    update_g=True,
                )
                conf_thresh = thresh_controller.get_thresh_global()

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar(
                    "train/loss_s", (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters
                )
                writer.add_scalar("train/loss_kd", loss_u_w_kd.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss KD: {:.3f}, Mask ratio: {:.3f}".format(
                        i,
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_s.avg,
                        total_loss_kd.avg,
                        total_mask_ratio.avg
                    )
                )

        model.eval()

        dice_class = [0] * 3

        with torch.no_grad():
            for img, mask in valloader:
                img, mask = img.cuda(), mask.cuda()

                h, w = img.shape[-2:]
                img = F.interpolate(
                    img,
                    (cfg["crop_size"], cfg["crop_size"]),
                    mode="bilinear",
                    align_corners=False,
                )

                img = img.permute(1, 0, 2, 3)

                pred = model(img)

                pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
                pred = pred.argmax(dim=1).unsqueeze(0)

                for cls in range(1, cfg["nclass"]):
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls - 1] += 2.0 * inter / union

        dice_class = [dice * 100.0 / len(valloader) for dice in dice_class]
        mean_dice = sum(dice_class) / len(dice_class)

        if rank == 0:
            for cls_idx, dice in enumerate(dice_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [{:} {:}] Dice: "
                    "{:.2f}".format(cls_idx, CLASSES[cfg["dataset"]][cls_idx], dice)
                )
            logger.info(
                "***** Evaluation ***** >>>> MeanDice: {:.2f}\n".format(mean_dice)
            )

            writer.add_scalar("eval/MeanDice", mean_dice, epoch)
            for i, dice in enumerate(dice_class):
                writer.add_scalar(
                    "eval/%s_dice" % (CLASSES[cfg["dataset"]][i]), dice, epoch
                )

        is_best = mean_dice > previous_best
        previous_best = max(mean_dice, previous_best)
        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))


if __name__ == "__main__":
    main()
