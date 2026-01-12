import os
from typing import Any, Dict, Optional, Callable

import torch
from absl.testing.parameterized import parameters
from lightning import LightningModule
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel


class LitMultimodalModule(LightningModule):
    """
    학습은 Lightning, 서빙 아티팩트는 Transformers save_pretrained로 생성
    """

    def __init__(
        self,
        model: PreTrainedModel,  # PreTrainedModel
        processor: Optional[Any] = None,  # ProcessorMixin
        optimizer: Optional[Callable[[], Optimizer]] = None,
        scheduler: Optional[Callable[[], LRScheduler]] = None,
        loss_func: nn.Module = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model", "processors", "loss_func"])

        self.hf_model = model
        self.processor = processor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func

    def forward(self, **inputs: torch.Tensor):
        return self.hf_model(**inputs, return_dict=True)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(**batch)
        labels = batch["labels"]
        loss = self.loss_func(outputs.logits, labels)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(labels))
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self.forward(**batch)
        labels = batch["labels"]
        loss = self.loss_func(outputs.logits, labels)
        self.log("valid/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(labels))
        return loss

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        outputs = self.forward(**batch)
        return outputs.logits

    # TODO: save pretrained
    def on_train_end(self) -> None:
        pass

    def configure_optimizers(self):
        if self.optimizer:
            optimizer = self.optimizer(params=self.hf_model.parameters())
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)

        if not self.scheduler:
            return optimizer
        else:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
