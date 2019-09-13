import pydoc

import torch
from torch.nn import DataParallel

from .data import make_data


class Metrics:
    def __init__(self, functions):
        self.functions = functions
        self.best_score = float('inf')
        self.best_epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}


class Factory:
    def __init__(self, params: dict):
        self.params = params

    def make_model(self, device) -> torch.nn.Module:
        model_name = self.params['model']
        model = pydoc.locate(model_name)(**self.params['model_params'])
        if isinstance(self.params.get('weights', None), str):
            model.load_state_dict(torch.load(self.params['weights']))
        return DataParallel(model).to(device)

    @staticmethod
    def make_optimizer(model: torch.nn.Module, stage: dict) -> torch.optim.Optimizer:
        return getattr(torch.optim, stage['optimizer'])(params=model.parameters(), **stage['optimizer_params'])

    @staticmethod
    def make_scheduler(optimizer, stage):
        return getattr(torch.optim.lr_scheduler, stage['scheduler'])(optimizer=optimizer, **stage['scheduler_params'])

    def make_loss(self, device) -> torch.nn.Module:
        loss = pydoc.locate(self.params['loss'])(**self.params['loss_params'])
        return loss.to(device)

    @staticmethod
    def get_metric_name(metric):
        return metric.split('.')[-1]

    def make_metrics(self) -> Metrics:
        return Metrics(
            {
                self.get_metric_name(metric): pydoc.locate(metric)(**params)
                for metric, params in self.params['metrics'].items()
            }
        )


class DataFactory:
    def __init__(self, params: dict, fold):
        self.fold = fold
        self.params = params

    def make_train_loader(self):
        return make_data(**self.params, mode='train', fold=self.fold)

    def make_val_loader(self):
        return make_data(**self.params, mode='val', fold=self.fold)
