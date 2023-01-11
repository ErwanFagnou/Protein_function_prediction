import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import ProteinDataset
from models.GNN import GNN

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Building model...")
    model = GNN(
        num_node_features=ProteinDataset.NUM_NODE_FEATURES - 3,  # removed the 3 coordinates from the
        num_classes=ProteinDataset.NUM_CLASSES,
    ).to(device)
    config = model.config

    print("Building dataset...")
    protein_dataset = ProteinDataset(
        batch_size=config.batch_size,
        num_validation_samples=config.num_validation_samples,
    )

    # wandb_logger = WandbLogger(project="ALTeGraD Kaggle challenge", entity="efagnou", name=config.name)
    # wandb_logger.log_hyperparams(config)
    #
    # save_dir = f"checkpoints/{wandb_logger.name}-{wandb_logger.version}"
    # val_checkpoint_callback = ModelCheckpoint(
    #     dirpath=save_dir,
    #     monitor="val_loss",
    #     filename="{epoch:02d}-{step:05d}-{val_loss:.4f}",
    # )
    # last_checkpoint_callback = ModelCheckpoint(
    #     dirpath=save_dir,
    #     filename="{epoch:02d}-{step:05d}-last",
    # )
    #
    model_path = get_unique_file_path("trained_models", f"{model.config.name}", "pt")
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")

    save_predictions(model, protein_dataset)

    trainer = Trainer(
        max_epochs=config.epochs,
        # logger=wandb_logger,
        # callbacks=[val_checkpoint_callback, last_checkpoint_callback, scheduler_callback],
        **trainer_kwargs,
    )
    trainer.fit(model, protein_dataset.train_loader, protein_dataset.val_loader)

    trainer.test(model, protein_dataset.test_loader)