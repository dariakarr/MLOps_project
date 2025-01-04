import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import nn
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class PneumoniaDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = datasets.ImageFolder(
                root=f"{self.data_dir}/train", transform=self.transform
            )
            self.val_dataset = datasets.ImageFolder(
                root=f"{self.data_dir}/val", transform=self.transform
            )
        if stage in ("test", None):
            self.test_dataset = datasets.ImageFolder(
                root=f"{self.data_dir}/test", transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class PneumoniaModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = torch.hub.load(
            "pytorch/vision:v0.13.1", "resnet50", pretrained=True
        )
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


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
