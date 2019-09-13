import random

import numpy as np
import torch
import yaml


def set_global_seeds(i: int):
    torch.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    random.seed(i)
    np.random.seed(i)


def get_config(path: str) -> dict:
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config


def batch2device(data, device):
    return {k: v if not hasattr(v, 'to') else v.to(device) for k, v in data.items()}


def mask2rle(img, size):
    width, height = size
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return ' '.join(rle)
