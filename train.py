import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import ProteinDataset


def train(model, protein_dataset, device, device_id, pretrained_seq_encoder=None):
    config = model.config

    wandb_logger = WandbLogger(project="ALTeGraD Kaggle challenge", entity="erwan-denis", name=config.name, group=model.experiment_name,
                               tags=['valid'])
    wandb_logger.log_hyperparams(config)
    if pretrained_seq_encoder is not None:
        wandb_logger.log_hyperparams(dict(pretrained_seq_encoder_name=pretrained_seq_encoder.config.name))

    scheduler_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    # swa_callback = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=15, swa_lrs=5e-4, device=device)

    save_dir = f"checkpoints/{wandb_logger.name}-{wandb_logger.version}"
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="{epoch:02d}-{step:05d}-last",
    )

    callbacks = [scheduler_callback, last_checkpoint_callback]  # , swa_callback]

    if config.num_validation_samples > 0:
        val_checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,
            monitor="val_loss",
            filename="{epoch:02d}-{step:05d}-{val_loss:.4f}",
        )
        callbacks.append(val_checkpoint_callback)

    trainer_kwargs = {}
    if device.type == 'cuda':
        trainer_kwargs['accelerator'] = 'gpu'
        # trainer_kwargs['devices'] = [max(range(torch.cuda.device_count()),
        #                                  key=lambda i: torch.cuda.get_device_properties(i).total_memory)]
        trainer_kwargs['devices'] = [device_id]

    trainer = Trainer(
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches if hasattr(config, 'accumulate_grad_batches') else None,
        #gradient_clip_val=100.0,
        logger=wandb_logger,
        callbacks=callbacks,
        **trainer_kwargs,
    )
    trainer.fit(model, protein_dataset.train_loader, protein_dataset.val_loader)

    wandb_logger.experiment.finish(quiet=True)
