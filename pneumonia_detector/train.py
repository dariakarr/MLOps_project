import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    from data_module import PneumoniaDataModule
    from model_module import PneumoniaModel

    # параметры из Hydra
    data_dir = cfg.data.data_dir
    batch_size = cfg.data.batch_size

    lr = cfg.model.lr

    max_epochs = cfg.train.max_epochs
    accelerator = cfg.train.accelerator

    data_module = PneumoniaDataModule(data_dir=data_dir, batch_size=batch_size)

    model = PneumoniaModel(lr=lr)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    logger = TensorBoardLogger(
        save_dir=cfg.logging.save_dir,  # путь из hydra
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        accelerator=accelerator,
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
