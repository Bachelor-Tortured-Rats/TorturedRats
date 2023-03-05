from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.data import CacheDataset, DataLoader

from src.utils.data_transformations import Addd


debug_transforms = Compose(
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

def load_IRCAD_dataset(ircad_path="/zhome/a2/4/155672/Desktop/Bachelor/3Dircadb1", patients_val=[1,4,5,6,7,8,9,17]):
    """Loads the IRCAD dataset from folder

    Args:
        ircad_path (str, optional): file path to 3Dircadb1. Defaults to "/zhome/a2/4/155672/Desktop/Bachelor/3Dircadb1".
        patients_val (list, optional): which patients to includes (defaults all). Defaults to [1,4,5,6,7,8,9,17].

    Returns:
        val_loader: data_loader  
    """
    # Defines data loaders
    train_images = [f'{ircad_path}/3Dircadb1.{i}/PATIENT_DICOM/' for i in patients_val]
    train_venoussystem = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/venoussystem/' for i in patients_val]
    train_artery = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/artery/' for i in patients_val]
    train_mask = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/liver/' for i in patients_val]
    val_files = [{"image": image_name, "label": label_name, "mask": mask_name, "label2": label2_name} for image_name, label_name, mask_name, label2_name in zip(train_images, train_venoussystem, train_mask, train_artery)]
    
    val_ds = CacheDataset(data=val_files, transform=debug_transforms, cache_rate=1.0, num_workers=4)    
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    return val_loader

