from collections import defaultdict
from typing import Dict

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .utils import batch2device

tqdm.monitor_interval = 0


class Runner:
    def __init__(self, factory, device, callbacks=None, stages: Dict[str, dict] = None):
        self.factory = factory
        self.device = device
        self.stages = stages

        self._model = None
        self._loss = None
        self._metrics = None

        self.current_stage = None
        self.current_stage_name = None
        self.global_epoch = 0
        self.optimizer = None
        self.scheduler = None

        self.callbacks = callbacks
        if callbacks is not None:
            self.callbacks.set_runner(self)

    @property
    def model(self):
        if self._model is None:
            self._model = self.factory.make_model(device=self.device)
        return self._model

    @property
    def loss(self):
        if self._loss is None:
            self._loss = self.factory.make_loss(device=self.device)
        return self._loss

    @property
    def metrics(self):
        if self._metrics is None:
            self._metrics = self.factory.make_metrics()
        return self._metrics

    def fit(self, data_factory):
        self.callbacks.on_train_begin()
        for stage_name, stage in self.stages.items():
            self.current_stage = stage
            self.current_stage_name = stage_name

            train_loader = data_factory.make_train_loader()
            val_loader = data_factory.make_val_loader()
            self.optimizer = self.factory.make_optimizer(self.model, stage)
            self.scheduler = self.factory.make_scheduler(self.optimizer, stage)

            self.callbacks.on_stage_begin()
            self._run_one_stage(train_loader, val_loader)
            self.callbacks.on_stage_end()
            torch.cuda.empty_cache()
        self.callbacks.on_train_end()

    def _run_one_stage(self, train_loader, val_loader):
        for epoch in range(self.current_stage['epochs']):
            train_loader.sampler.set_epoch(epoch)
            print(f'positive ratio: {train_loader.sampler.positive_ratio}')
            self.callbacks.on_epoch_begin(self.global_epoch)

            self.model.train()
            self.metrics.train_metrics = self._run_one_epoch(epoch, train_loader, is_train=True)

            self.model.eval()
            self.metrics.val_metrics = self._run_one_epoch(epoch, val_loader, is_train=False)
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(self.metrics.val_metrics['dice'], epoch)
            else:
                self.scheduler.step(epoch)
            self.callbacks.on_epoch_end(self.global_epoch)
            self.global_epoch += 1

    def _run_one_epoch(self, epoch: int, loader, is_train: bool = True) -> Dict[str, float]:
        epoch_report = defaultdict(float)
        progress_bar = tqdm(
            iterable=enumerate(loader),
            total=len(loader),
            desc=f"Epoch {epoch} {['validation', 'train'][is_train]}ing...",
            ncols=0
        )
        metrics = {}
        with torch.set_grad_enabled(is_train):
            for i, data in progress_bar:
                self.callbacks.on_batch_begin(i)
                step_report = self._make_step(data, is_train)
                for key, value in step_report.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    epoch_report[key] += value

                metrics = {k: v / (i + 1) for k, v in epoch_report.items()}
                progress_bar.set_postfix(**{k: f'{v:.5f}' for k, v in metrics.items()})
                self.callbacks.on_batch_end(i, step_report=step_report, is_train=is_train)
        return metrics

    def _make_step(self, data: Dict[str, torch.Tensor], is_train: bool) -> Dict[str, float]:
        report = {}
        data = self.batch2device(data)
        images = data['image']
        labels = data['mask']
        non_empty_labels = data['non_empty']
        predictions, empty_predictions = self.model(images)
        loss = self.loss(predictions, labels, empty_predictions, non_empty_labels)
        report['loss'] = loss.data

        if is_train:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            report['grad'] = grad_norm
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            for metric, f in self.metrics.functions.items():
                report[metric] = f(torch.sigmoid(predictions).cpu().numpy(), labels.cpu().numpy())
        return report

    def batch2device(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return batch2device(data, device=self.device)
