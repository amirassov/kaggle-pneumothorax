import argparse
import os
import os.path as osp
from collections import defaultdict
from glob import glob

import mmcv
import torch
from tqdm import tqdm

from src.data import make_data
from src.factory import Factory
from src.inference import PytorchInference
from src.utils import get_config, set_global_seeds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--path', type=str)
    parser.add_argument('--fold', type=str)
    parser.add_argument('--output', type=str)
    return parser.parse_args()


def collect_weight_paths(path, fold):
    return sorted(glob(osp.join(path, f'fold{fold}', '*.pth')))


def main():
    args = parse_args()
    set_global_seeds(666)
    config = get_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runner = PytorchInference(device)

    test_loader = make_data(**config['data_params'], mode='test')
    weights_paths = collect_weight_paths(args.path, args.fold)

    os.makedirs(args.output, exist_ok=True)
    predictions = defaultdict(lambda: {'mask': 0, 'empty': 0})
    for i, weights_path in enumerate(weights_paths):
        torch.cuda.empty_cache()
        print(f'weights: {weights_path}')
        config['train_params']['weights'] = weights_path
        factory = Factory(config['train_params'])
        model = factory.make_model(device)
        for result in tqdm(
            iterable=runner.predict(model, test_loader), total=len(test_loader) * config['data_params']['batch_size']
        ):
            ensemble_mask = (predictions[result['image_id']]['mask'] * i + result['mask']) / (i + 1)
            ensemble_empty = (predictions[result['image_id']]['empty'] * i + result['empty']) / (i + 1)
            predictions[result['image_id']]['mask'] = ensemble_mask
            predictions[result['image_id']]['empty'] = ensemble_empty
    mmcv.dump(dict(predictions), osp.join(args.output, f'fold_{args.fold}.pkl'))


if __name__ == '__main__':
    main()
