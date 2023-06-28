import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torchvision.models import resnet50, ResNet50_Weights
from torcheval.metrics import ReciprocalRank
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class FungiDataset(Dataset):
    def __init__(self, root_dir, metadata_path, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(metadata_path)
        df["image_path"] = df["image_path"].str.replace("JPG", "jpg")
        self.images = df["image_path"].values.tolist()
        self.labels = df["class_id"].values.tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.root_dir + self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class FungiDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(232),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(45),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        self.val_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(232),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage=None):
        self.train_dataset = FungiDataset(
            root_dir="DF20-300px/DF20_300/",
            metadata_path="DF20-train_metadata.csv",
            transform=self.train_transform,
        )
        self.val_dataset = FungiDataset(
            root_dir="DF20-300px/DF20_300/",
            metadata_path="DF20-val_metadata.csv",
            transform=self.val_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


class Resnet50(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet50.fc = torch.nn.Linear(in_features=2048, out_features=out_features)

    def forward(self, x):
        return self.resnet50(x)


class ModeloBase(LightningModule):
    def __init__(
        self,
        lr,
        model,
        criterion,
        species_count,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.lr = lr
        self.model = model
        self.criterion = criterion
        self.species_count = species_count

        metrics = MetricCollection(
            {
                "Accuracy": MulticlassAccuracy(
                    num_classes=species_count, average="micro"
                ),
                "BalancedAccuracy": MulticlassAccuracy(num_classes=species_count),
                "F1Score": MulticlassF1Score(num_classes=species_count),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.train_mrr = ReciprocalRank()
        self.val_mrr = ReciprocalRank()

    def forward(self, x):
        return self.model(x)

    def loss(self, preds, ys):
        return self.criterion(preds, ys)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4, eps=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, verbose=True
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.train_mrr.update(y_hat, y)

        y_hat = y_hat.argmax(dim=-1)
        self.train_metrics.update(y_hat, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss, "train/labels": y, "train/predictions": y_hat}

    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
        self.log(
            "train/MRR", self.train_mrr.compute().mean(), on_step=False, on_epoch=True
        )
        self.train_metrics.reset()
        self.train_mrr.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.val_mrr.update(y_hat, y)

        y_hat = y_hat.argmax(dim=-1)
        self.val_metrics.update(y_hat, y)

        self.log("val/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss, "val/labels": y, "val/predictions": y_hat}

    def validation_epoch_end(self, outputs):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True)
        self.log("val/MRR", self.val_mrr.compute().mean(), on_step=False, on_epoch=True)
        self.val_metrics.reset()
        self.val_mrr.reset()


import wandb
from pytorch_lightning.loggers import WandbLogger

batch_size = 256
num_workers = 10
species_count = 1604
criterion = torch.nn.CrossEntropyLoss()
id = None
lr = 0.0003

datamodule = FungiDataModule(batch_size=batch_size, num_workers=num_workers)
datamodule.setup()

if id is None:
    id = wandb.util.generate_id()
print(id)


module = Resnet50(out_features=species_count)

model = ModeloBase(
    lr=lr,
    model=module,
    criterion=criterion,
    species_count=species_count,
)

wandb_logger = WandbLogger(project="FungiCLEF2022", id=id, resume="allow")

from pytorch_lightning.callbacks import LearningRateMonitor

trainer = Trainer(
    logger=wandb_logger,
    accelerator="gpu",
    devices=-1,
    max_epochs=30,
    precision=16,
    callbacks=[LearningRateMonitor(logging_interval="epoch")],
)

trainer.fit(model, datamodule=datamodule)
wandb.finish()

"""
trainer = Trainer(accelerator="gpu", devices=-1, max_epochs=30, precision=16)

# Run learning rate finder
lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule, num_training=1000)

# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()
print(new_lr)

# Plot with
fig = lr_finder.plot(suggest=True)
fig.show()
"""
