from copy import deepcopy
import os
import random
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from torchio import transforms as tiot


def obtain_cutmix_box(
    img_size=(112, 112, 80),
    p=0.5,
    size_min=0.02,
    size_max=0.4,
    ratio_1=0.3,
    ratio_2=1 / 0.3,
):
    mask = torch.zeros(img_size)
    img_size_x, img_size_y, img_size_z = img_size

    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size_x * img_size_y * img_size_z
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.power(size / ratio, 1 / 3))
        cutmix_h = int(np.power(size * ratio, 1 / 3))
        cutmix_d = int(np.power(size, 1 / 3))
        x = np.random.randint(0, img_size_x - cutmix_w)
        y = np.random.randint(0, img_size_y - cutmix_h)
        z = np.random.randint(0, img_size_z - cutmix_d)

        if (
            x + cutmix_w <= img_size_x
            and y + cutmix_h <= img_size_y
            and z + cutmix_d <= img_size_z
        ):
            break

    mask[z : z + cutmix_d, y : y + cutmix_h, x : x + cutmix_w] = 1

    return mask


class LAHeart(Dataset):
    """LA Dataset"""

    def __init__(
        self, base_dir=None, mode="train", num=None, transform=None, id_path=None
    ):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.mode = mode

        train_path = self._base_dir + "/train.list"
        test_path = self._base_dir + "/test.list"

        if id_path is not None:
            with open(id_path, "r") as f:
                self.image_list = f.readlines()
        else:
            if mode == "train":
                with open(train_path, "r") as f:
                    self.image_list = f.readlines()
            elif mode == "test":
                with open(test_path, "r") as f:
                    self.image_list = f.readlines()

        self.image_list = [item.replace("\n", "") for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list * 21
            self.image_list = self.image_list[:num]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/" + image_name + "/mri_norm2.h5", "r")

        image = h5f["image"][:]
        label = h5f["label"][:]

        sample = {"image": image, "label": label}

        if self.mode == "val":
            return (image).float(), (label).long()

        sample_auged = self.transform(sample)
        image, label = sample_auged["image"], sample_auged["label"]

        if self.mode == "train_l" or self.mode == "test":
            return (image).float(), (label).long()

        img = deepcopy(image)
        img_s1, img_s2 = deepcopy(image), deepcopy(image)

        if random.random() < 0.8:
            img_s1 = tiot.RandomBiasField()(img_s1)
            img_s1 = tiot.RandomGamma((-0.5, 0.5))(img_s1)

        if random.random() < 0.5:
            img_s1 = tiot.RandomBlur((0.1, 2))(img_s1)

        if random.random() < 0.8:
            img_s2 = tiot.RandomBiasField()(img_s2)
            img_s2 = tiot.RandomGamma((-0.5, 0.5))(img_s2)

        if random.random() < 0.5:
            img_s2 = tiot.RandomBlur((0.1, 2))(img_s2)

        return img, img_s1, img_s2, obtain_cutmix_box(p=0.5), obtain_cutmix_box(p=0.5)


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # pad the sample if necessary
        if (
            label.shape[0] <= self.output_size[0]
            or label.shape[1] <= self.output_size[1]
            or label.shape[2] <= self.output_size[2]
        ):
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(
                image,
                [(pw, pw), (ph, ph), (pd, pd)],
                mode="constant",
                constant_values=0,
            )
            label = np.pad(
                label,
                [(pw, pw), (ph, ph), (pd, pd)],
                mode="constant",
                constant_values=0,
            )

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.0))
        h1 = int(round((h - self.output_size[1]) / 2.0))
        d1 = int(round((d - self.output_size[2]) / 2.0))

        label = label[
            w1 : w1 + self.output_size[0],
            h1 : h1 + self.output_size[1],
            d1 : d1 + self.output_size[2],
        ]
        image = image[
            w1 : w1 + self.output_size[0],
            h1 : h1 + self.output_size[1],
            d1 : d1 + self.output_size[2],
        ]

        return {"image": image, "label": label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if self.with_sdf:
            sdf = sample["sdf"]

        # pad the sample if necessary
        if (
            label.shape[0] <= self.output_size[0]
            or label.shape[1] <= self.output_size[1]
            or label.shape[2] <= self.output_size[2]
        ):
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(
                image,
                [(pw, pw), (ph, ph), (pd, pd)],
                mode="constant",
                constant_values=0,
            )
            label = np.pad(
                label,
                [(pw, pw), (ph, ph), (pd, pd)],
                mode="constant",
                constant_values=0,
            )
            if self.with_sdf:
                sdf = np.pad(
                    sdf,
                    [(pw, pw), (ph, ph), (pd, pd)],
                    mode="constant",
                    constant_values=0,
                )

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[
            w1 : w1 + self.output_size[0],
            h1 : h1 + self.output_size[1],
            d1 : d1 + self.output_size[2],
        ]
        image = image[
            w1 : w1 + self.output_size[0],
            h1 : h1 + self.output_size[1],
            d1 : d1 + self.output_size[2],
        ]
        if self.with_sdf:
            sdf = sdf[
                w1 : w1 + self.output_size[0],
                h1 : h1 + self.output_size[1],
                d1 : d1 + self.output_size[2],
            ]
            return {"image": image, "label": label, "sdf": sdf}
        else:
            return {"image": image, "label": label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {"image": image, "label": label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        noise = np.clip(
            self.sigma
            * np.random.randn(image.shape[0], image.shape[1], image.shape[2]),
            -2 * self.sigma,
            2 * self.sigma,
        )
        noise = noise + self.mu
        image = image + noise
        return {"image": image, "label": label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]),
            dtype=np.float32,
        )
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {"image": image, "label": label, "onehot_label": onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample["image"]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(
            np.float32
        )
        if "onehot_label" in sample:
            return {
                "image": torch.from_numpy(image),
                "label": torch.from_numpy(sample["label"]).long(),
                "onehot_label": torch.from_numpy(sample["onehot_label"]).long(),
            }
        else:
            return {
                "image": torch.from_numpy(image),
                "label": torch.from_numpy(sample["label"]).long(),
            }


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(
        self, primary_indices, secondary_indices, batch_size, secondary_batch_size
    ):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
