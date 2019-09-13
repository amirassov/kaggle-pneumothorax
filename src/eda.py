import argparse

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np

from src.data import PneumothoraxDataset, make_filenames
from src.utils import get_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the model config')
    parser.add_argument('--output', type=str, help='Path to the annotation output')
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config(args.config)
    transform = A.load(config['data_params']['transform']['train'], 'yaml')
    transform.transforms = transform.transforms[:-1]
    img_filenames, mask_filenames, non_emptiness = make_filenames(
        data_folder=config['data_params']['data_folder'],
        mode='train',
        fold='0',
        folds_path=config['data_params']['folds_path']
    )
    dataset = PneumothoraxDataset(img_filenames, mask_filenames, transform=transform, non_emptiness=non_emptiness)
    for sample in dataset:
        image = sample['image']
        mask = sample['mask']
        plt.figure(figsize=(20, 10))
        plt.imshow(np.hstack([image, np.stack([mask, mask, mask], axis=-1)]))
        plt.show()


if __name__ == '__main__':
    main()
