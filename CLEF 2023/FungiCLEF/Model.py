import os
from typing import List, Tuple
import numpy as np

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    convnext_base,
    ConvNeXt_Base_Weights,
    vit_b_16,
    ViT_B_16_Weights,
    efficientnet_b7,
    EfficientNet_B7_Weights,
    convnext_large,
    ConvNeXt_Large_Weights,
    resnet152,
    ResNet152_Weights,
)
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torcheval.metrics import ReciprocalRank
import wandb
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision("high")


class FungiDataset(Dataset):
    def __init__(self, root_dir, metadata_path, lower_image_path=False, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        df = pd.read_csv(metadata_path)
        if lower_image_path:
            df["image_path"] = df["image_path"].str.replace("JPG", "jpg")
        self.images = df["image_path"].values.tolist()

        # Para manejar el -1 como 0 hay que mover todas a la derecha
        df["class_id"] = df["class_id"].apply(lambda x: x + 1)

        self.labels = df["class_id"].values.tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.root_dir + self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class FungiDatasetAll(Dataset):
    def __init__(
        self,
        root_dir_train,
        root_dir_val,
        metadata_path_train,
        metadata_path_val,
        transform=None,
    ):
        super().__init__()
        self.transform = transform

        df_train = pd.read_csv(metadata_path_train)
        df_train["image_path"] = df_train["image_path"].str.lower()
        df_train["image_path"] = df_train["image_path"].apply(
            lambda x: os.path.join(root_dir_train, x)
        )

        df_val = pd.read_csv(metadata_path_val)
        df_val["image_path"] = df_val["image_path"].apply(
            lambda x: os.path.join(root_dir_val, x)
        )

        df = pd.concat([df_train, df_val], ignore_index=True)

        self.images = df["image_path"].tolist()
        self.labels = df["class_id"].apply(lambda x: x + 1).tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class FungiDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(232),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(232),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, stage=None):
        self.train_dataset = FungiDataset(
            root_dir="DF20_300/",
            metadata_path="FungiCLEF2023_train_metadata_PRODUCTION.csv",
            lower_image_path=True,
            transform=self.train_transform,
        )
        self.val_dataset = FungiDataset(
            root_dir="DF21_300/",
            metadata_path="FungiCLEF2023_val_metadata_PRODUCTION.csv",
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

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


class FungiDataModuleAll(FungiDataModule):
    def setup(self, stage=None):
        train_dataset = FungiDatasetAll(
            "DF20_300/",
            "DF21_300/",
            "FungiCLEF2023_train_metadata_PRODUCTION.csv",
            "FungiCLEF2023_val_metadata_PRODUCTION.csv",
            transform=self.train_transform,
        )
        val_dataset = FungiDatasetAll(
            "DF20_300/",
            "DF21_300/",
            "FungiCLEF2023_train_metadata_PRODUCTION.csv",
            "FungiCLEF2023_val_metadata_PRODUCTION.csv",
            transform=self.val_transform,
        )

        train_indices, test_indices = train_test_split(
            range(len(train_dataset)), test_size=0.05, stratify=train_dataset.labels
        )

        train_labels = np.array(train_dataset.labels)[train_indices]
        self.class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )

        self.train_dataset = Subset(train_dataset, train_indices)
        self.val_dataset = Subset(val_dataset, test_indices)


class Resnet50(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet50.fc = torch.nn.Linear(in_features=2048, out_features=out_features)

    def forward(self, x):
        return self.resnet50(x)


class Resnet152(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.resnet152 = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        print(self.resnet152)
        self.resnet152.fc = torch.nn.Linear(in_features=2048, out_features=out_features)

    def forward(self, x):
        return self.resnet152(x)


class Convnext(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        self.convnext.classifier[2] = nn.Linear(1024, out_features, bias=True)

    def forward(self, x):
        return self.convnext(x)


class ConvnextL(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.convnext = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
        self.convnext.classifier[2] = nn.Linear(1536, out_features, bias=True)

    def forward(self, x):
        return self.convnext(x)


class VitBase(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.vit_b_16 = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit_b_16.heads.head = nn.Linear(768, out_features, bias=True)

    def forward(self, x):
        return self.vit_b_16(x)


class EfficientNetB7(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.efficientnet_b7 = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
        print(self.efficientnet_b7)
        self.efficientnet_b7.classifier[1] = torch.nn.Linear(
            in_features=2560, out_features=out_features
        )

    def forward(self, x):
        return self.efficientnet_b7(x)


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


class BaseModel(LightningModule):
    def __init__(
        self,
        lr: float,
        model,
        criterion,
        species_count: int,
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
        self.metrics = {
            "train": metrics.clone(prefix="train/"),
            "val": metrics.clone(prefix="val/"),
        }
        self.mrr = {"train": ReciprocalRank(), "val": ReciprocalRank()}

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

    def step(self, batch, stage: str):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.mrr[stage].update(y_hat, y)

        y_hat = y_hat.argmax(dim=-1)
        self.metrics[stage].update(y_hat, y)

        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss, f"{stage}/labels": y, f"{stage}/predictions": y_hat}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def epoch_end(self, outputs, stage: str):
        self.log_dict(self.metrics[stage].compute(), on_step=False, on_epoch=True)
        self.log(
            f"{stage}/MRR",
            self.mrr[stage].compute().mean(),
            on_step=False,
            on_epoch=True,
        )
        self.metrics[stage].reset()
        self.mrr[stage].reset()

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, "val")


class ModeloEnsamblado(LightningModule):
    def __init__(
        self,
        lr,
        models,
        criterion,
        species_count,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.lr = lr
        self.models = nn.ModuleList(models)
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
        self.test_metrics = metrics.clone(prefix="test/")

        self.test_mrr = ReciprocalRank()

    def forward(self, x):
        logits_list = [model(x) for model in self.models]
        logits = torch.stack(logits_list, dim=1).mean(dim=1)
        return logits

    def loss(self, preds, ys):
        return self.criterion(preds, ys)

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.test_mrr.update(y_hat, y)

        y_hat = y_hat.argmax(dim=-1)
        self.test_metrics.update(y_hat, y)

        self.log("test/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss, "test/labels": y, "test/predictions": y_hat}

    def test_epoch_end(self, outputs):
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        self.log(
            "test/MRR", self.test_mrr.compute().mean(), on_step=False, on_epoch=True
        )
        self.test_metrics.reset()
        self.test_mrr.reset()


batch_size = 128
num_workers = 10
species_count = 1604 + 1

id = None
lr = 0.0003
# Probar después 0.0007870457896950985

datamodule = FungiDataModuleAll(batch_size=batch_size, num_workers=num_workers)
datamodule.setup()

# criterion = torch.nn.CrossEntropyLoss(
#    weight=torch.tensor(datamodule.class_weights, dtype=torch.float)
# )

criterion = torch.nn.CrossEntropyLoss()

if id is None:
    id = wandb.util.generate_id()
print(id)

"""
module = Resnet152(out_features=species_count)

model = ModeloBase(
    lr=lr,
    model=module,
    criterion=criterion,
    species_count=species_count,
)

wandb_logger = WandbLogger(project="FungiCLEF2023", id=id, resume="allow")

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

module1 = ModeloBase.load_from_checkpoint(
    "/home/ruben/Documents/CLEF 2023/FungiCLEF/FungiCLEF2023/pkfvzrtc/checkpoints/epoch=29-step=79440.ckpt",
    lr=lr,
    model=Convnext(out_features=species_count),
    criterion=torch.nn.CrossEntropyLoss(),
    species_count=species_count,
).model

module2 = ModeloBase.load_from_checkpoint(
    "/home/ruben/Documents/CLEF 2023/FungiCLEF/FungiCLEF2023/fav2fjf4/checkpoints/epoch=29-step=158880.ckpt",
    lr=lr,
    model=EfficientNetB7(out_features=species_count),
    criterion=torch.nn.CrossEntropyLoss(),
    species_count=species_count,
).model

model = ModeloEnsamblado(
    lr=lr,
    models=[module1, module2],
    criterion=criterion,
    species_count=species_count,
)

wandb_logger = WandbLogger(project="FungiCLEF2023", id=id, resume="allow")

trainer = Trainer(
    logger=wandb_logger,
    accelerator="gpu",
    devices=-1,
    max_epochs=0,
    num_sanity_val_steps=0,
    precision=16,
)

trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
wandb.finish()
