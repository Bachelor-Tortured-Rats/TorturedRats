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
                              SaveImaged, ScaleIntensityRanged, Spacingd, RandSpatialCropSamplesd)
from sklearn.model_selection import KFold, train_test_split

from src.utils.data_transformations import selectPatchesd, RandSelectPatchesd


def select_label(x):
    return x != 0


# Train transforms to use for self supervised learning
# on the hepatic dataset
transforms_3drpl = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        LabelFilterd(keys=["label"], applied_labels=[1]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-58,
            a_max=478,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(
            keys=["image", "label"], select_fn=select_label, source_key="label", margin=10),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(
            0.8, 0.8, 2.5), mode=("bilinear", "nearest")),
        RandZoomd(keys=["image", "label"], prob=0.2,
                  min_zoom=1, max_zoom=1.5, mode=['area', 'nearest']),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.1,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.05,
            prob=0.2,
        ),
        RandSpatialCropSamplesd(keys = ["image"], roi_size = (54,54,30), random_size = False, num_samples = 8),
        RandSelectPatchesd(keys=["image"]) # This one makes a random offset from the middle
    ]
)

train_transforms_aug = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        LabelFilterd(keys=["label"], applied_labels=[1]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-58,
            a_max=478,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(
            keys=["image", "label"], select_fn=select_label, source_key="label", margin=10),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(
            0.8, 0.8, 2.5), mode=("bilinear", "nearest")),
        RandZoomd(keys=["image", "label"], prob=0.2,
                  min_zoom=1, max_zoom=1.5, mode=['area', 'nearest']),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.1,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.05,
            prob=0.2,
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            # spatial_size=(96, 96, 96),
            spatial_size=(48, 48, 48),
            pos=1,
            neg=1,
            num_samples=8,
            image_key="image",
            image_threshold=0,
        ),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        LabelFilterd(keys=["label"], applied_labels=[1]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-58,
            a_max=478,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(
            keys=["image", "label"], select_fn=select_label, source_key="label", margin=10),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(
            0.8, 0.8, 2.5), mode=("bilinear", "nearest")),
    ]
)


def load_hepatic_dataset(data_dir, k_fold,numkfold=5, train_label_proportion=-1, batch_size=1, setup='default'):
    train_images = sorted(
        glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))

    data_dicts = [{"image": image_name, "label": label_name}
                  for image_name, label_name in zip(train_images, train_labels)]

    if numkfold != 1:
        kf = KFold(n_splits=numkfold, shuffle=True, random_state=420)
        kf_splits = kf.split(data_dicts)
        train_index, test_index = list(kf_splits)[k_fold]

        # only use a subset of the training data,
        if train_label_proportion != -1:
            train_index = train_index[:int(len(train_index)*train_label_proportion)] # only use a subset of the training data,

        # split train into train and val
        train_index = train_index[:-max(int(len(train_index)*.2),1)]
        val_index = train_index[-max(int(len(train_index)*.2),1):]
    
        # split data into train and val files
        train_files = [data_dicts[i] for i in train_index]
        val_files = [data_dicts[i] for i in val_index]
        test_files = [data_dicts[i] for i in test_index]


    else: # only train data if numkfold == 1
        train_files, test_files = train_test_split(data_dicts,test_size=0.1, random_state=420, shuffle=True)

    if setup == 'transfer' or setup == 'random' or setup=='3drpl':
        train_ds = CacheDataset(data=train_files, transform=train_transforms_aug, cache_rate=1.0, num_workers=None)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=None)
        test_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=1.0, num_workers=None)  # do not validate on augmented data
    elif setup == '3drpl_pretask':
        assert numkfold == 1, '3drpl_pretask only works with numkfold=1'
        train_ds = CacheDataset(data=train_files, transform=transforms_3drpl, cache_rate=1.0, num_workers=None)
        test_ds = CacheDataset(data=test_files, transform=transforms_3drpl, cache_rate=1.0, num_workers=None)
        val_ds = None
    else:  # default
        raise NotImplementedError('Setup not implemented')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)
    
    # sets val_loader to None if no validation set
    if val_ds:
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
    else:
        val_loader = None

    return train_loader, val_loader, test_loader
