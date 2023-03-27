from monai.networks.nets import UNet
import torch
from monai.networks.layers import Norm
from src.utils.models import UNetEnc
from src.models.unet_model import create_unet


def load_unet_enc(model_path: str, device,
              spatial_dims=3,
              in_channels=1,
              out_channels=2,
              channels=(16, 32, 64, 128, 256),
              strides=(2, 2, 2, 2),
              num_res_units=2,
              ):
    """Loads a trained unet model

    Args:
        model_path (str): path to model

    Returns:
        model: a loaded pytorch unet model
    """
    checkpoint = torch.load(model_path, map_location=device)
    params = {
        "spatial_dims": checkpoint['spatial_dims'],
        "in_channels": checkpoint['in_channels'],
        "out_channels": checkpoint['out_channels'],
        "channels": checkpoint['channels'],
        "strides": checkpoint['strides'],
        "num_res_units": checkpoint['num_res_units'],
        "dropout": checkpoint.get('dropout', 0),
        "kernel_size": checkpoint.get('kernel_size', 3)
    }
    
    model, params = create_unet(device=device, **params)

    ## Fixed version of partial loading from Adam Paszke (apaszke) @ pytorch.org 
    ## https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3

    pretrained_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    return model, params


def create_unet_enc(device,
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                dropout=0,
                kernel_size=3
                ):
    """Create UNet with encoder only

    Args:
        model_path (str): path to model

    Returns:
        model: a loaded pytorch unet model
    """
    params = {
        "spatial_dims": spatial_dims,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "channels": channels,
        "strides": strides,
        "num_res_units": num_res_units,
        "dropout": dropout,
        "kernel_size": kernel_size,
        "up_kernel_size": kernel_size
    }
    model = UNetEnc(norm=Norm.BATCH, **params)
    model.to(device)

    return model, params
