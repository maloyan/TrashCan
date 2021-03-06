import os

import albumentations as A
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from pkg_resources import resource_filename
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from trash.dataset import TargetDataset
from trash.engine import eval_fn, train_fn

config = OmegaConf.load(resource_filename(__name__, "configs/config.yaml"))

wandb.init(
    config=config,
    project=config["project"],
    name=f"{config['image_size']}_{config['model']}",
)

transforms_train = A.Compose(
    [
        A.Resize(height=config["image_size"], width=config["image_size"], p=1),
        A.OneOf(
            [A.HorizontalFlip(p=0.7), A.VerticalFlip(p=0.2), A.RandomRotate90(p=0.1),],
            p=0.7,
        ),
        A.OneOf(
            [
                A.GaussNoise(p=0.5),
                A.RandomGamma(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
            ],
            p=0.3,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.2, rotate_limit=180, p=0.2
        ),
    ],
    p=1,
)

transforms_test = A.Compose(
    [A.Resize(height=config["image_size"], width=config["image_size"], p=1)], p=1,
)

df = pd.read_csv(config["data_csv"])
df_train, df_val = train_test_split(df)


train_data = [
    os.path.join(config["data_path"], "train", i) for i in df_train.ID_img.values
]
train_target = df_train["class"].values

valid_data = [
    os.path.join(config["data_path"], "train", i) for i in df_val.ID_img.values
]
valid_target = df_val["class"].values

train_dataset = TargetDataset(
    train_data,
    train_target,
    is_test=False,
    augmentation=transforms_train,
    classes=config["classes"],
)
valid_dataset = TargetDataset(
    valid_data,
    valid_target,
    is_test=False,
    augmentation=transforms_test,
    classes=config["classes"],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    drop_last=True,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    drop_last=False,
)

model = timm.create_model(
    config["model"], num_classes=config["classes"], pretrained=True
)
print("PARALLEL")
model = torch.nn.DataParallel(model, device_ids=config["device_ids"])

criterion = F.binary_cross_entropy_with_logits

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    factor=config["decay"],
    patience=config["patience"],
    verbose=True,
)

best_loss = 1000
for _ in range(config["epochs"]):
    train_loss = train_fn(train_loader, model, optimizer, criterion, config["device"])
    val_loss, metric = eval_fn(valid_loader, model, criterion, config["device"])

    scheduler.step(val_loss)

    if val_loss < best_loss:
        print("Model saved!")
        best_loss = val_loss
        torch.save(
            model.module,
            f"{config['checkpoints']}/{config['image_size']}_{config['model']}.pt",
        )
    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_metric": metric})
