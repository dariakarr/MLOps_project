from pneumonia_detector.data_module import PneumoniaDataModule
from pneumonia_detector.model_module import PneumoniaModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import nn
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def main():
    data_module = PneumoniaDataModule(data_dir="./data", batch_size=8)
    model = PneumoniaModel(lr=1e-4)
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[checkpoint_callback],
        gpus=0,  # 1 for gpu
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
