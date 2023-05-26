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
                              SaveImaged, ScaleIntensityRanged, Spacingd,RandSpatialCropd,RandSpatialCropSamplesd)
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from src.utils.data_transformations import selectPatchesd, RandSelectPatchesd,RandSelectPatchesLarged


def select_kidney(x):
    return x == 1

transforms_3drpl = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        CropForegroundd(keys=["image","mask"], source_key="mask"), 
        Spacingd(keys=["image", "mask"], pixdim=(0.0226, 0.0226, 0.0226), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        RandSpatialCropSamplesd(keys=["image"], roi_size=(300, 300, 300), random_size=False, num_samples=8),
        RandSelectPatchesLarged(keys=["image"]) # This one makes a random offset from the middle
    ]
)

train_transforms_rat_kidney_segmented = Compose(
    [
        LoadImaged(keys=["image", "label", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask",'label']),
        Orientationd(keys=["image", "label",'mask'], axcodes="RAS"),
        CropForegroundd(keys=["image","label"], source_key="mask"), 
        Spacingd(keys=["image", "label"], pixdim=(0.0226, 0.0226, 0.0226), mode=("bilinear", "nearest")),
        ScaleIntensityRanged( # combines all labels to label 1
            keys=["label"],
            a_min=0,
            a_max=1,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], select_fn=select_kidney, source_key="label", margin=10),# needs to be after ScaleIntensityRanged
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-58,
            a_max=478,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=8,
            image_key="image",
            image_threshold=0,
        ),
    ]
)

val_transforms_rat_kidney_segmented = Compose(
    [
        LoadImaged(keys=["image", "label", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask",'label']),
        Orientationd(keys=["image", "label",'mask'], axcodes="RAS"),
        CropForegroundd(keys=["image","label"], source_key="mask"), 
        Spacingd(keys=["image", "label"], pixdim=(0.0226, 0.0226, 0.0226), mode=("bilinear", "nearest")),
        ScaleIntensityRanged( # combines all labels to label 1
            keys=["label"],
            a_min=0,
            a_max=1,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], select_fn=select_kidney, source_key="label", margin=10),# needs to be after ScaleIntensityRanged
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=256,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
)


test_transforms = Compose([LoadImaged(
    keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),])


def load_ray_kidney_dataset(data_dir, k_fold,numkfold=5, train_label_proportion=-1, batch_size=1, setup='default'):
    raise NotImplementedError()

def get_loader_rat_kidney_full(data_dir,setup,batch_size=1,num_samples=16):
      # Rats files that have masks
    rats = [21, 22, 24,25,28,33,36,37,38,43,47,48,51,52,55,57][:num_samples]
    
    train_images = [f'{data_dir}/aligned/rat{i}_aligned_rigid.nii' for i in rats]
    train_masks = [f'{data_dir}/maskKidney/rat{i}_kidneyMaskProc.nii.gz' for i in rats]

    data_dicts = [{"image": image_name, "mask": train_mask}
                  for image_name, train_mask in zip(train_images, train_masks)]

    train_files, val_files = data_dicts[:-1], data_dicts[-1:]

    if setup == '3drpl_pretask':
        train_ds = CacheDataset(data=train_files, transform=transforms_3drpl, cache_rate=1, num_workers=None)
        val_ds = CacheDataset(data=val_files, transform=transforms_3drpl, cache_rate=1, num_workers=None) 
    else: 
        raise NotImplementedError("setup not implemented")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    return train_loader, val_loader


def get_rat_kidney_segmented(data_dir,batch_size=1):
    '''
    Returns a dataloader for the rat kidney dataset subset that contains segmentation masks
    This is only a part of rat 37
    '''

    train_images = [f'{data_dir}/analysis_rat37/rat37_reorient.nii.gz']
    train_labels = [f'{data_dir}/analysis_rat37/vessel_zoom_ground_truth-ish_rat37.nii.gz']
    train_masks = [f'{data_dir}/study_diabetic/aligned/rat37_aligned_rigid.nii']

    data_dicts = [{"image": image_name, "mask": train_mask, "label": train_label} for image_name, train_mask, train_label in zip(train_images, train_masks, train_labels)]

    train_files, val_files = data_dicts, data_dicts

    train_ds = CacheDataset(data=train_files, transform=train_transforms_rat_kidney_segmented, cache_rate=1.0, num_workers=None)
    val_ds = CacheDataset(data=val_files, transform=val_transforms_rat_kidney_segmented, cache_rate=1.0, num_workers=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    return train_loader, val_loader
