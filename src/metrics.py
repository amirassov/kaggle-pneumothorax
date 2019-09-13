import numpy as np

EPS = 1e-4


class DiceMetric:
    def __init__(self, score_threshold=0.5):
        self.score_threshold = score_threshold

    def __call__(self, predictions, gt):
        mask = predictions > self.score_threshold
        batch_size = mask.shape[0]

        mask = mask.reshape(batch_size, -1).astype(np.bool)
        gt = gt.reshape(batch_size, -1).astype(np.bool)

        intersection = np.logical_and(mask, gt).sum(axis=1)
        union = mask.sum(axis=1) + gt.sum(axis=1) + EPS
        loss = (2.0 * intersection + EPS) / union
        return loss.mean()
