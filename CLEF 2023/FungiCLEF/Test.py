import os
import pandas as pd
import csv
import torch
from PIL import Image
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
)
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torcheval.metrics import ReciprocalRank

torch.set_float32_matmul_precision("high")


class FungiDatasetTest(Dataset):
    def __init__(
        self,
        root_dir,
        metadata_path,
        transform=None,
    ):
        super().__init__()
        self.transform = transform

        df = pd.read_csv(metadata_path)
        df["image_path"] = df["image_path"].apply(lambda x: os.path.join(root_dir, x))
        self.images = df["image_path"].tolist()
        self.obs = df["observationID"].tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        ob = self.obs[idx]
        if self.transform:
            image = self.transform(image)
        return image, ob


class FungiDataModuleTest(LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.Resize(232),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, stage=None):
        self.dataset = FungiDatasetTest(
            root_dir="DF21_300/",
            metadata_path="FungiCLEF2023_public_test_metadata_PRODUCTION.csv",
            transform=self.transform,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


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
            optimizer, T_max=50, verbose=True
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


class TestEngineFungi(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1)
        return idx - 1

    def on_test_epoch_start(self):
        self.probs_tensor = torch.zeros(0, dtype=torch.float, device="cpu")
        self.obs_tensor = torch.zeros(0, dtype=torch.short, device="cpu")

    def on_test_epoch_end(self):
        with open(f"results.csv", "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(["observation_id", "class_id"])
            for ob in torch.unique(self.obs_tensor):
                indices = (self.obs_tensor == ob).nonzero(as_tuple=True)[0]
                probs = torch.index_select(self.probs_tensor, 0, indices)
                idx = torch.argmax(torch.mean(probs, dim=0)).item()
                writer.writerow([ob.item(), idx - 1])

    def test_step(self, batch, batch_idx):
        x, obs = batch
        logits = self.model(x)
        preds = self.softmax(logits)
        # x, obs = batch
        # preds = self.model(x, None)
        self.probs_tensor = torch.cat([self.probs_tensor, preds.cpu()])
        self.obs_tensor = torch.cat([self.obs_tensor, obs.cpu()])


class TestEngineEnsembleFungi(LightningModule):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits_list = [model(x) for model in self.models]
        logits = torch.stack(logits_list, dim=1).mean(dim=1)
        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1)
        return idx - 1

    def on_test_epoch_start(self):
        self.probs_tensor = torch.zeros(0, dtype=torch.float, device="cpu")
        self.obs_tensor = torch.zeros(0, dtype=torch.short, device="cpu")

    def on_test_epoch_end(self):
        with open(f"results.csv", "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(["observation_id", "class_id"])
            for ob in torch.unique(self.obs_tensor):
                indices = (self.obs_tensor == ob).nonzero(as_tuple=True)[0]
                probs = torch.index_select(self.probs_tensor, 0, indices)
                idx = torch.argmax(torch.mean(probs, dim=0)).item()
                writer.writerow([ob.item(), idx - 1])

    def test_step(self, batch, batch_idx):
        x, obs = batch
        logits_list = [model(x) for model in self.models]
        logits = torch.stack(logits_list, dim=1).mean(dim=1)
        preds = self.softmax(logits)
        # x, obs = batch
        # preds = self.model(x, None)
        self.probs_tensor = torch.cat([self.probs_tensor, preds.cpu()])
        self.obs_tensor = torch.cat([self.obs_tensor, obs.cpu()])


class Convnext(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        self.convnext.classifier[2] = nn.Linear(1024, out_features, bias=True)

    def forward(self, x):
        return self.convnext(x)


class VitBase(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.vit_b_16 = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        print(self.vit_b_16)
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


batch_size = 128
num_workers = 10
species_count = 1604 + 1

datamodule = FungiDataModuleTest(batch_size=batch_size, num_workers=num_workers)
"""
module = EfficientNetB7(out_features=species_count)
module = ModeloBase.load_from_checkpoint(
    # "/home/ruben/Documents/CLEF 2023/FungiCLEF/FungiCLEF2023/pzqpf8nx/checkpoints/epoch=29-step=79440.ckpt",
    "/home/ruben/Documents/CLEF 2023/FungiCLEF/FungiCLEF2023/fav2fjf4/checkpoints/epoch=29-step=158880.ckpt",
    lr=0.0003,
    model=module,
    # criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(0.0, dtype=torch.float)),
    criterion=torch.nn.CrossEntropyLoss(),
    batch_size=batch_size,
    species_count=species_count,
).model

module = TestEngineFungi(module)
"""
module1 = ModeloBase.load_from_checkpoint(
    "/home/ruben/Documents/CLEF 2023/FungiCLEF/FungiCLEF2023/pkfvzrtc/checkpoints/epoch=29-step=79440.ckpt",
    lr=0.0003,
    model=Convnext(out_features=species_count),
    criterion=torch.nn.CrossEntropyLoss(),
    species_count=species_count,
).model

module2 = ModeloBase.load_from_checkpoint(
    "/home/ruben/Documents/CLEF 2023/FungiCLEF/FungiCLEF2023/fav2fjf4/checkpoints/epoch=29-step=158880.ckpt",
    lr=0.0003,
    model=EfficientNetB7(out_features=species_count),
    criterion=torch.nn.CrossEntropyLoss(),
    species_count=species_count,
).model

module = TestEngineEnsembleFungi([module1, module2])


trainer = Trainer(accelerator="gpu", devices=-1, max_epochs=50, precision=16)

trainer.test(module, datamodule=datamodule)

filepath = "model.onnx"
input_sample = torch.randn(1, 3, 224, 224)
module.to_onnx(filepath, input_sample, export_params=True)
