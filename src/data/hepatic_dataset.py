import glob
import os
import pdb

import numpy as np
from monai.data import CacheDataset, DataLoader
from monai.transforms import (AsDiscrete, AsDiscreted, Compose,
                              CropForegroundd, EnsureChannelFirstd, Invertd,
                              LabelFilterd, LoadImaged, Orientationd,
                              Rand3DElasticd, RandCropByPosNegLabeld,
                              RandRotate90d, RandShiftIntensityd, RandZoomd,
                              SaveImaged, ScaleIntensityRanged, Spacingd)
from sklearn.model_selection import KFold

from src.utils.data_transformations import selectPatchesd


def select_kidney(x):
    return x == 1


# Train transforms to use for self supervised learning
# on the hepatic dataset
transforms_3drpl = Compose(
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
        CropForegroundd(
            keys=["image", "label"], select_fn=select_kidney, source_key="label", margin=20),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(48, 48, 24),
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        selectPatchesd(keys=["image"]),
    ]
)


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
        CropForegroundd(
            keys=["image", "label"], select_fn=select_kidney, source_key="label", margin=20),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
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

        # user can also add other random transforms
        # RandAffined(
        #     keys=['image', 'label'],
        #     mode=('bilinear', 'nearest'),
        #     prob=1.0, spatial_size=(96, 96, 96),
        #     rotate_range=(0, 0, np.pi/15),
        #     scale_range=(0.1, 0.1, 0.1)),
    ]
)
train_transforms_aug = Compose(
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
        CropForegroundd(
            keys=["image", "label"], select_fn=select_kidney, source_key="label", margin=20),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        RandZoomd(keys=["image", "label"], prob=0.3,
                  min_zoom=1.3, max_zoom=1.5, mode=['area', 'nearest']),
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(10, 10),
            magnitude_range=(300, 300),
            prob=0.1,
            padding_mode='zeros',
            mode=['bilinear', 'nearest']),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.1,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.2,
        ),
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

        # user can also add other random transforms
        # RandAffined(
        #     keys=['image', 'label'],
        #     mode=('bilinear', 'nearest'),
        #     prob=1.0, spatial_size=(96, 96, 96),
        #     rotate_range=(0, 0, np.pi/15),
        #     scale_range=(0.1, 0.1, 0.1)),
    ]
)
val_transforms = Compose(
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
        CropForegroundd(
            keys=["image", "label"], select_fn=select_kidney, source_key="label", margin=20),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ]
)
val_transforms_aug = Compose(
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
        CropForegroundd(
            keys=["image", "label"], select_fn=select_kidney, source_key="label", margin=20),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        RandZoomd(keys=["image", "label"], prob=0.3,
                  min_zoom=1.3, max_zoom=1.5, mode=['area', 'nearest']),
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(10, 10),
            magnitude_range=(300, 300),
            prob=0.1,
            padding_mode='zeros',
            mode=['bilinear', 'nearest']),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.1,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.2,
        ),
    ]
)
test_transforms = Compose([LoadImaged(
    keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),])


def load_hepatic_dataset(data_dir, k_fold,numkfold=5, train_label_proportion=-1, batch_size=1, setup='default'):
    train_images = sorted(
        glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

    data_dicts = [{"image": image_name, "label": label_name}
                  for image_name, label_name in zip(train_images, train_labels)]


    kf = KFold(n_splits=numkfold, shuffle=True, random_state=420)
    kf_splits = kf.split(data_dicts)
    train_index, val_index = list(kf_splits)[k_fold]

    train_files = [data_dicts[i] for i in train_index]
    val_files = [data_dicts[i] for i in val_index]

    if train_label_proportion != -1:
        train_files = train_files[:int(
            len(train_files)*train_label_proportion)]

    if setup == 'transfer' or setup == 'random':
        train_ds = CacheDataset(data=train_files, transform=train_transforms_aug, cache_rate=1.0, num_workers=0)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)  # do not validate on augmented data
    elif setup == '3drpl_pretask':
        train_ds = CacheDataset(data=train_files, transform=transforms_3drpl, cache_rate=1.0, num_workers=0)
        val_ds = CacheDataset(data=val_files, transform=transforms_3drpl, cache_rate=1.0, num_workers=0)
    else:  # default
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=0)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    return train_loader, val_loader
