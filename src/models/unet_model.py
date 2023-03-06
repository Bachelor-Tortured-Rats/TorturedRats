from monai.networks.nets import UNet
import torch
from monai.networks.layers import Norm

def load_unet(model_path: str, device,
              spatial_dims  = 3,
              in_channels   = 1,
              out_channels  = 2,
              channels      = (16, 32, 64, 128, 256),
              strides       = (2, 2, 2, 2),
              num_res_units = 2,
              ):
    """Loads a trained unet model

    Args:
        model_path (str): path to model

    Returns:
        model: a loaded pytorch unet model
    """
    checkpoint = torch.load(model_path, map_location=device)

    model = create_unet(device=device,
        spatial_dims    = checkpoint['spatial_dims'],
        in_channels     = checkpoint['in_channels'],
        out_channels    = checkpoint['out_channels'],
        channels        = checkpoint['channels'],
        strides         = checkpoint['strides'],
        num_res_units   = checkpoint['num_res_units']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def create_unet(device,
              spatial_dims  = 3,
              in_channels   = 1,
              out_channels  = 2,
              channels      = (16, 32, 64, 128, 256),
              strides       = (2, 2, 2, 2),
              num_res_units = 2,
              ):
    """Loads a trained unet model

    Args:
        model_path (str): path to model

    Returns:
        model: a loaded pytorch unet model
    """
    model = UNet(
        spatial_dims    = spatial_dims,
        in_channels     = in_channels,
        out_channels    = out_channels,
        channels        = channels,
        strides         = strides,
        num_res_units   = num_res_units,
        norm            = Norm.BATCH,
    )
    model.to(device)

    return model
