import argparse
from typing import Dict

import cv2
import mmcv
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import mask2rle

EMPTY = '-1'
FINAL_SIZE = (1024, 1024)


def parse_args():
    parser = argparse.ArgumentParser(description='submit')
    parser.add_argument('predictions', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--sample_submission', type=str)
    parser.add_argument('--empty_threshold', type=float, default=0.4)
    parser.add_argument('--empty_score_threshold', type=float, default=0.8)
    parser.add_argument('--area_threshold', type=float, default=800)
    parser.add_argument('--mask_score_threshold', type=float, default=0.4)
    return parser.parse_args()


def make_rle(
    predictions: Dict[str, Dict[str, np.ndarray]],
    image_ids: set,
    empty_threshold: float,
    area_threshold: float,
    empty_score_threshold: float,
    mask_score_threshold: float,
) -> Dict[str, str]:
    rle_predictions = {}
    for image_id, prediction in tqdm(predictions.items()):
        empty = prediction['empty']
        mask = prediction['mask']
        image_id = image_id.split('.png')[0]
        if image_id in image_ids:
            if empty < empty_threshold:
                rle_predictions[image_id] = EMPTY
            elif np.sum(mask > empty_score_threshold) < area_threshold:
                rle_predictions[image_id] = EMPTY
            else:
                mask = np.array(mask > mask_score_threshold).astype(np.uint8)
                mask = cv2.resize(src=mask * 255, dsize=FINAL_SIZE, interpolation=cv2.INTER_NEAREST)
                rle_predictions[image_id] = mask2rle(mask.T, FINAL_SIZE)
    return rle_predictions


def main():
    args = parse_args()
    sample_submission = pd.read_csv(args.sample_submission)
    image_ids = set(sample_submission['ImageId'].tolist())
    predictions = mmcv.load(args.predictions)

    rle_predictions = make_rle(
        predictions=predictions,
        image_ids=image_ids,
        empty_threshold=args.empty_threshold,
        empty_score_threshold=args.empty_score_threshold,
        area_threshold=args.area_threshold,
        mask_score_threshold=args.mask_score_threshold,
    )

    submission = pd.DataFrame(
        {
            'ImageId': list(rle_predictions.keys()),
            'EncodedPixels': list(rle_predictions.values())
        }
    )
    submission.loc[submission['EncodedPixels'] == '', 'EncodedPixels'] = EMPTY
    submission.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
