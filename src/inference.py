import numpy as np
import torch


class PytorchInference:
    def __init__(self, device):
        self.device = device

    @staticmethod
    def to_numpy(images):
        return images.data.cpu().numpy()

    @staticmethod
    def run_one_predict(model, images):
        predictions, emptiness = model(images)
        predictions = torch.sigmoid(predictions)
        emptiness = torch.sigmoid(emptiness)
        return predictions, emptiness

    @staticmethod
    def flip_tensor_lr(images):
        invert_indices = torch.arange(images.data.size()[-1] - 1, -1, -1).long()
        return images.index_select(3, invert_indices.cuda())

    def tta(self, model, images):
        predictions, emptiness = self.run_one_predict(model, images)
        predictions_lr, emptiness_lr = self.run_one_predict(model, self.flip_tensor_lr(images))
        predictions_lr = self.flip_tensor_lr(predictions_lr)
        predictions_tta = torch.stack([predictions, predictions_lr]).mean(0)
        emptiness_tta = torch.stack([emptiness, emptiness_lr]).mean(0)
        return predictions_tta, emptiness_tta

    def predict(self, model, loader):
        model = model.to(self.device).eval()
        with torch.no_grad():
            for data in loader:
                images = data['image'].to(self.device)
                predictions, emptiness = self.tta(model, images)
                image_ids = data['image_id']
                for prediction, empty, image_id in zip(predictions, emptiness, image_ids):
                    prediction = np.moveaxis(self.to_numpy(prediction), 0, -1)
                    empty = self.to_numpy(empty)
                    yield {'mask': prediction, 'empty': empty, 'image_id': image_id}
