import logging

import hydra

import torch
from hydra.utils import instantiate
from lightning import LightningModule, LightningDataModule, seed_everything, Trainer
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed), workers=True)

    # Load & Update Huggingface config to use in hydra style
    module = load_module(cfg)
    datamodule = load_datamodule(cfg)
    trainer = instantiate(cfg.trainer)

    # _debug_batch(datamodule)

    trainer.fit(model=module, datamodule=datamodule)

    # trainer.validate(module=module, datamodule=datamodule)
    log.info("Training finished!")


def _debug_batch(datamodule: LightningDataModule):
    valid_batch = next(iter(datamodule.val_dataloader()))
    for k, v in valid_batch.items():
        if isinstance(v, torch.Tensor):
            log.info(f"{k} ({type(v)}): {v.size()}")
        else:
            log.info(f"{k} ({type(v)}): {v[:5]}")


def load_module(cfg: DictConfig) -> LightningModule:
    hf_config = OmegaConf.to_container(cfg.hf_config)
    partial_config = instantiate(cfg.module.model.config)

    partial_model = instantiate(cfg.module.model)
    model = partial_model(config=partial_config(**hf_config))
    log.info(f"Current Model:\n{model}")
    partial_module = instantiate(cfg.module)
    module = partial_module(model=model)
    return module


def load_datamodule(cfg: DictConfig) -> LightningDataModule:
    return instantiate(cfg.datamodule)


if __name__ == "__main__":
    # 실행 예:
    #   PYTHONPATH=. python src/train.py
    main()
