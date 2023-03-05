from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    LabelFilterd,
    Lambdad,
)


from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import matplotlib.animation as animation
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform

set_determinism(seed=420)
    
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label",'mask','label2']),
        EnsureChannelFirstd(keys=["image", "label",'mask','label2']),
        Addd(keys=["label"],source_key='label2'),
        CropForegroundd(keys=["image", "label"], source_key="mask"), # crops the scan to the size of the nyre
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ScaleIntensityRanged(
            keys=["label"],
            a_min=0,
            a_max=1,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),   
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            # spatial_size=(96, 96, 96),
            spatial_size=(48, 48, 48), # needs to be divisible by 2*4=
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        
    ]
)

# Defines transformations
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label",'mask','label2']),
        EnsureChannelFirstd(keys=["image", "label",'mask','label2']),
        Addd(keys=["label"],source_key='label2'),
        CropForegroundd(keys=["image", "label"], source_key="mask"), # crops the scan to the size of the nyre
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ScaleIntensityRanged(
            keys=["label"],
            a_min=0,
            a_max=1,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ]
)


# Defines data loaders
patients_val = [1,4,4,5,6,7,8,9,17] # patients to include
train_images = [f'/zhome/a2/4/155672/Desktop/Bachelor/3Dircadb1/3Dircadb1.{i}/PATIENT_DICOM/' for i in patients_val]
train_venoussystem = [f'/zhome/a2/4/155672/Desktop/Bachelor/3Dircadb1/3Dircadb1.{i}/MASKS_DICOM/venoussystem/' for i in patients_val]
train_artery = [f'/zhome/a2/4/155672/Desktop/Bachelor/3Dircadb1/3Dircadb1.{i}/MASKS_DICOM/artery/' for i in patients_val]
train_mask = [f'/zhome/a2/4/155672/Desktop/Bachelor/3Dircadb1/3Dircadb1.{i}/MASKS_DICOM/liver/' for i in patients_val]
data_dicts = [{"image": image_name, "label": label_name, "mask": mask_name, "label2": label2_name} for image_name, label_name, mask_name, label2_name in zip(train_images, train_venoussystem, train_mask, train_artery)]
train_files, val_files = data_dicts[:-3], data_dicts[-3:] #split of train and test


train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=1, num_workers=4)


val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

# Creates models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
)
model.to(device)
model.load_state_dict(torch.load('models/best_metric_model.pth',map_location=device))


loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
dice_metric = DiceMetric(include_background=False, reduction="mean")

max_epochs = 20
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )

                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(),  "models/IRCAD_finetuned_best_metric_model.pth")
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )


print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.savefig('figures/training_graphs.png')