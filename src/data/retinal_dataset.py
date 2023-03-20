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
    Rand3DElasticd,
    RandRotate90d,
    RandShiftIntensityd,
    RandZoomd
)
from monai.data import CacheDataset, DataLoader

import os
import glob
import pdb
import numpy as np

from src.utils.data_transformations import selectPatchesd


def select_kidney(x):
    return x == 1

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        LabelFilterd(keys=["label"], applied_labels=[1]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], select_fn=select_kidney, source_key="label", margin=20),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            # spatial_size=(96, 96, 96),
            spatial_size=(48, 48, 48),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        selectPatchesd(keys=["image", "label"], source_key="label"),

    ]
)

def load_hepatic_dataset(data_dir,test_train_split=.8,train_label_proportion=-1,aug=False):
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    train_files, _ = data_dicts[:int(len(data_dicts)*test_train_split)], data_dicts[int(len(data_dicts)*test_train_split):]
    
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

    return train_loader, _


