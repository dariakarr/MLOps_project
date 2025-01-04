import pytorch_lightning as pl
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
