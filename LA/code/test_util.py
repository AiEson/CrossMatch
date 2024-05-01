import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import pandas as pd
from collections import OrderedDict


def test_all_case(
    net,
    image_list,
    num_classes,
    patch_size=(112, 112, 80),
    stride_xy=18,
    stride_z=4,
    save_result=True,
    test_save_path=None,
    preproc_fn=None,
):
    total_metric = 0.0
    metric_dict = OrderedDict()
    metric_dict["name"] = list()
    metric_dict["dice"] = list()
    metric_dict["jaccard"] = list()
    metric_dict["asd"] = list()
    metric_dict["95hd"] = list()
    for image_path in tqdm(image_list):
        case_name = image_path.split("/")[-2]
        id = image_path.split("/")[-1]
        h5f = h5py.File(image_path, "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes
        )

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            metric_dict["name"].append(case_name)
            metric_dict["dice"].append(single_metric[0])
            metric_dict["jaccard"].append(single_metric[1])
            metric_dict["asd"].append(single_metric[2])
            metric_dict["95hd"].append(single_metric[3])
            # print(metric_dict)

        total_metric += np.asarray(single_metric)

        if save_result:
            test_save_path_temp = os.path.join(test_save_path, case_name)
            if not os.path.exists(test_save_path_temp):
                os.makedirs(test_save_path_temp)
            nib.save(
                nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                test_save_path_temp + "/" + id + "_pred.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)),
                test_save_path_temp + "/" + id + "_img.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)),
                test_save_path_temp + "/" + id + "_gt.nii.gz",
            )
    avg_metric = total_metric / len(image_list)
    if save_result:
        metric_csv = pd.DataFrame(metric_dict)
        metric_csv.to_csv(test_save_path + "/metric.csv", index=False)
    print("average metric is {}".format(avg_metric))

    return avg_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(
            image,
            [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
            mode="constant",
            constant_values=0,
        )
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ]
                test_patch = np.expand_dims(
                    np.expand_dims(test_patch, axis=0), axis=0
                ).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[
                    :,
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] = (
                    score_map[
                        :,
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ]
                    + y
                )
                cnt[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] = (
                    cnt[
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ]
                    + 1
                )
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[
            wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d
        ]
        score_map = score_map[
            :, wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d
        ]
    return label_map, score_map


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = prediction == i
        label_tmp = label == i
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = (
            2
            * np.sum(prediction_tmp * label_tmp)
            / (np.sum(prediction_tmp) + np.sum(label_tmp))
        )
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
