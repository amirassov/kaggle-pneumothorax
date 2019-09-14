import argparse
import os.path as osp
from collections import defaultdict
from glob import glob
from typing import Dict

import mmcv
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='ensemble')
    parser.add_argument('path_pattern', type=str)
    parser.add_argument('--output', type=str)
    return parser.parse_args()


def make_ensemble(path_pattern: str) -> Dict[str, Dict[str, np.ndarray]]:
    ensemble_predictions = defaultdict(lambda: {'mask': 0, 'empty': 0})
    for i, path in enumerate(tqdm(sorted(glob(path_pattern)))):
        print(f'load {path}')
        predictions = mmcv.load(path)
        for image_id, prediction in predictions.items():
            mask_ensemble = (ensemble_predictions[image_id]['mask'] * i + prediction['mask']) / (i + 1)
            empty_ensemble = (ensemble_predictions[image_id]['empty'] * i + prediction['empty']) / (i + 1)
            ensemble_predictions[image_id]['mask'] = mask_ensemble
            ensemble_predictions[image_id]['empty'] = empty_ensemble
    return dict(ensemble_predictions)


def main():
    args = parse_args()
    ensemble_predictions = make_ensemble(args.path_pattern)
    mmcv.dump(dict(ensemble_predictions), osp.join(args.output, f'ensemble.pkl'))


if __name__ == '__main__':
    main()
