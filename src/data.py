import os.path as osp
from glob import glob
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler


class PneumothoraxDataset(Dataset):
    def __init__(self, img_filenames: List[str], mask_filenames: List[str], non_emptiness: np.ndarray, transform=None):
        self.img_filenames = img_filenames
        self.mask_filenames = mask_filenames
        self.non_emptiness = non_emptiness
        self.transform = transform

    def __getitem__(self, index):
        sample = {'image': cv2.imread(self.img_filenames[index], 1)}
        if self.mask_filenames is not None:
            mask = cv2.imread(self.mask_filenames[index], 0)
            if mask is not None:
                sample['mask'] = mask
            else:
                assert not self.non_emptiness[index]
                sample['mask'] = np.zeros(sample['image'].shape[:2])

        if self.non_emptiness is not None:
            sample['non_empty'] = float(self.non_emptiness[index])

        if self.transform is not None:
            sample = self.transform(**sample)

        sample['image_id'] = osp.basename(self.img_filenames[index])
        return sample

    def __len__(self):
        return len(self.img_filenames)


class EmptySampler(Sampler):
    def __init__(self, data_source: PneumothoraxDataset, positive_ratio_range: Tuple[float, float], epochs: int = 50):
        super().__init__(data_source)
        assert len(positive_ratio_range) == 2
        self.positive_indices = np.where(data_source.non_emptiness == 1)[0]
        self.negative_indices = np.where(data_source.non_emptiness == 0)[0]
        self.positive_ratio_range = positive_ratio_range
        self.positive_num: int = len(self.positive_indices)
        self.current_epoch: int = 0
        self.epochs: int = epochs

    @property
    def positive_ratio(self) -> float:
        np.random.seed(self.current_epoch)
        min_ratio, max_ratio = self.positive_ratio_range
        return max_ratio - (max_ratio - min_ratio) / self.epochs * self.current_epoch

    @property
    def negative_num(self) -> int:
        assert self.positive_ratio <= 1.0
        return int(self.positive_num // self.positive_ratio - self.positive_num)

    def __iter__(self):
        negative_indices = np.random.choice(self.negative_indices, size=self.negative_num)
        indices = np.random.permutation(np.hstack((negative_indices, self.positive_indices)))
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.positive_num + self.negative_num

    def set_epoch(self, epoch):
        self.current_epoch = epoch


def make_filenames(data_folder, mode, fold=None, folds_path=None):
    if mode == 'test':
        img_filenames = sorted(glob(osp.join(data_folder, 'test', '*')))
        return img_filenames, None, None

    folds = pd.read_csv(folds_path)
    if mode == 'train':
        folds = folds[folds['fold'].astype(str) != str(fold)]
    elif mode == 'val':
        folds = folds[folds['fold'].astype(str) == str(fold)]
    img_filenames = folds['ImageId'].apply(lambda x: osp.join(data_folder, 'train', x)).tolist()
    mask_filenames = folds['ImageId'].apply(lambda x: osp.join(data_folder, 'mask', x)).tolist()
    non_emptiness = folds['non_empty'].astype(bool).values
    return img_filenames, mask_filenames, non_emptiness


def make_data_loader(dataset, sampler, batch_size, num_workers):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=num_workers, pin_memory=False
    )


def make_data(
    data_folder: str,
    mode: str,
    transform: dict,
    num_workers: int,
    batch_size: int,
    fold: str = None,
    folds_path: str = None,
    positive_ratio_range: Tuple[float, float] = (0.3, 0.8),
    epochs: int = 50
):
    img_filenames, mask_filenames, non_emptiness = make_filenames(
        data_folder=data_folder, mode=mode, fold=fold, folds_path=folds_path
    )
    _transform = A.load(transform[mode], 'yaml')
    _transform.transforms = _transform.transforms + [ToTensor()]
    dataset = PneumothoraxDataset(
        img_filenames=img_filenames, mask_filenames=mask_filenames, transform=_transform, non_emptiness=non_emptiness
    )

    sampler = EmptySampler(data_source=dataset, positive_ratio_range=positive_ratio_range, epochs=epochs)
    loader = make_data_loader(
        dataset=dataset, sampler=sampler if mode == 'train' else None, batch_size=batch_size, num_workers=num_workers
    )
    return loader
