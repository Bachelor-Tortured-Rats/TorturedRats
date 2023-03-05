from monai.utils import set_determinism
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
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader
from monai.config import KeysCollection
from monai.transforms import MapTransform

import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import argparse

from src.utils.data_transformations import Addd
from src.visualization.plot_functions import animate_CT

def load_IRCAD_dataset(ircad_path="/zhome/a2/4/155672/Desktop/Bachelor/3Dircadb1", patients_val=[1,4,5,6,7,8,9,17]):
    """Loads the IRCAD dataset from folder

    Args:
        ircad_path (str, optional): file path to 3Dircadb1. Defaults to "/zhome/a2/4/155672/Desktop/Bachelor/3Dircadb1".
        patients_val (list, optional): which patients to includes (defaults all). Defaults to [1,4,5,6,7,8,9,17].

    Returns:
        val_loader: data_loader  
    """
    data_transforms = Compose(
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
    train_images = [f'{ircad_path}/3Dircadb1.{i}/PATIENT_DICOM/' for i in patients_val]
    train_venoussystem = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/venoussystem/' for i in patients_val]
    train_artery = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/artery/' for i in patients_val]
    train_mask = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/liver/' for i in patients_val]
    val_files = [{"image": image_name, "label": label_name, "mask": mask_name, "label2": label2_name} for image_name, label_name, mask_name, label2_name in zip(train_images, train_venoussystem, train_mask, train_artery)]
    
    val_ds = CacheDataset(data=val_files, transform=data_transforms, cache_rate=1.0, num_workers=4)    
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    return val_loader

def load_unet(model_path: str,device):
    """Loads a trained unet model

    Args:
        model_path (str): path to model

    Returns:
        model: a loaded pytorch unet model
    """
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

    model.load_state_dict(torch.load(model_path,map_location=device))
    return model


def eval_model(model, val_loader, device, saving_path="report/figures/validator", show_inference=False,save_animation=False):
    """Evaluates a unet pytorch model

    Args:
        model (pytorch): pytorch unet model
        val_loader (data_loader): data loader
        saving_path (str, optional): folder to save images. Defaults to "figures/validator".
        show_inference (bool, optional): save image of inference. Defaults to False.
        save_animation (bool, optional): save animation of inference. Defaults to False.
    """

    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            
            val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)

            dice_metric(y_pred=torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :,:], y=val_data["label"][0, 0, :, :,:])
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            print('dice score ', metric)
            # reset the status for next validation round
            dice_metric.reset()

            if save_animation:
                animate_CT(240*(val_data["image"][0, 0, :, :, :]+6)/12 , angle = 2, masks = [val_data["label"][0, 0, :, :,:], torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :,:]], alpha=0.2, filename="figures/validator/slices")
            
            # compute metric for current iteration
            
            # plot the slice [:, :, 80]'
            if show_inference:
                plt.figure(f"check dice score: {metric}", (18, 12))
                
                plt.subplot(1, 3, 1)
                plt.title(f"image {i} dice score: {metric}")
                plt.imshow(val_data["image"][0, 0, :, :, 60], cmap="gray")
                plt.subplot(1, 3, 2)
                plt.title(f"label {i}")
                plt.imshow(val_data["label"][0, 0, :, :, 60])
                plt.subplot(1, 3, 3)
                plt.title(f"output {i}")
                plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 60])
                
                plt.savefig(f'{saving_path}/validation_{i}.png')

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='data_path', type=str, help="A string argument")
    parser.add_argument(dest='model_path', type=str, help="An integer argument")
    parser.add_argument('--animation', action="store_true", default=False)
    parser.add_argument('--inference', action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    set_determinism(seed=420)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = arg_parser()

    val_loader = load_IRCAD_dataset(ircad_path=args.data_path,patients_val=[1,4])
    unet =  load_unet(model_path= args.model_path, device=device)
    eval_model(unet, val_loader, device,save_animation=args.animation,show_inference=args.inference)
