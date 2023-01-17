import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import ProteinDataset


def train(model, device, pretrained_seq_encoder=None, do_train=True):
    config = model.config

    model = model.to(device)
    if pretrained_seq_encoder is not None:
        print("Pretrained sequence encoder:", pretrained_seq_encoder)
        pretrained_seq_encoder = pretrained_seq_encoder.to(device)

    protein_dataset = ProteinDataset(
        batch_size=config.batch_size,
        num_validation_samples=config.num_validation_samples,
        pretrained_seq_encoder=pretrained_seq_encoder,
        transforms=model.transforms if hasattr(model, 'transforms') else None,
        pca_dim=model.PCA_DIM,
    )

    if not do_train:
        return protein_dataset

    wandb_logger = WandbLogger(project="ALTeGraD Kaggle challenge", entity="efagnou", name=config.name, group=model.experiment_name,
                               tags=['valid'])
    wandb_logger.log_hyperparams(config)
    if pretrained_seq_encoder is not None:
        wandb_logger.log_hyperparams(dict(pretrained_seq_encoder_name=pretrained_seq_encoder.config.name))

    save_dir = f"checkpoints/{wandb_logger.name}-{wandb_logger.version}"
    val_checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        monitor="val_loss",
        filename="{epoch:02d}-{step:05d}-{val_loss:.4f}",
    )
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="{epoch:02d}-{step:05d}-last",
    )

    scheduler_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    trainer_kwargs = {}
    if device.type == 'cuda':
        trainer_kwargs['accelerator'] = 'gpu'
        # trainer_kwargs['devices'] = [max(range(torch.cuda.device_count()),
        #                                  key=lambda i: torch.cuda.get_device_properties(i).total_memory)]
        trainer_kwargs['devices'] = [2]

    trainer = Trainer(
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches if hasattr(config, 'accumulate_grad_batches') else None,
        #gradient_clip_val=100.0,
        logger=wandb_logger,
        callbacks=[val_checkpoint_callback, last_checkpoint_callback, scheduler_callback],
        **trainer_kwargs,
    )
    trainer.fit(model, protein_dataset.train_loader, protein_dataset.val_loader)

    return protein_dataset
