from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    RandCropByPosNegLabeld,
)
from monai.data import CacheDataset, DataLoader

from src.utils.data_transformations import Addd

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

def load_IRCAD_dataset(ircad_path, train_patients=[5,6,7,8,9,17],val_patients=[1,4]):
    """Loads the IRCAD dataset from folder

    Args:
        ircad_path (str, optional): file path to 3Dircadb1. Defaults to "/zhome/a2/4/155672/Desktop/Bachelor/3Dircadb1".
        patients_val (list, optional): which patients to includes (defaults all). Defaults to [1,4,5,6,7,8,9,17].

    Returns:
        val_loader: data_loader  
    """
    # Defines data loaders
    train_images = [f'{ircad_path}/3Dircadb1.{i}/PATIENT_DICOM/' for i in train_patients]
    train_venoussystem = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/venoussystem/' for i in train_patients]
    train_artery = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/artery/' for i in train_patients]
    train_mask = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/liver/' for i in train_patients]
    train_files = [{"image": image_name, "label": label_name, "mask": mask_name, "label2": label2_name} for image_name, label_name, mask_name, label2_name in zip(train_images, train_venoussystem, train_mask, train_artery)]
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=1, num_workers=4)


    val_images = [f'{ircad_path}/3Dircadb1.{i}/PATIENT_DICOM/' for i in val_patients]
    val_venoussystem = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/venoussystem/' for i in val_patients]
    val_artery = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/artery/' for i in val_patients]
    val_mask = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/liver/' for i in val_patients]
    val_files = [{"image": image_name, "label": label_name, "mask": mask_name, "label2": label2_name} for image_name, label_name, mask_name, label2_name in zip(val_images, val_venoussystem, val_mask, val_artery)]
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)    
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    return train_loader, val_loader

