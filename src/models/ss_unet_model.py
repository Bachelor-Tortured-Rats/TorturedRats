from monai.networks.nets import UNet
import torch
from monai.networks.layers import Norm
from src.models.UNetEnc import UNetEnc
from src.models.unet_model import create_unet


def load_unet_enc(model_path: str, device):
    """Loads a trained unet model

    Args:
        model_path (str): path to encoder u-net model

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


def init_lr(model, encoder_lr, decoder_lr):
    ## Setting learning rate of encoder and decoder separately from Juan Montesinos (JuanFMontesinos) @ pytorch.org
    ## https://discuss.pytorch.org/t/how-to-set-a-different-learning-rate-for-a-single-layer-in-a-network/48552/10
    encoder_params_list = [x for x, param in model.named_parameters() if param.requires_grad and "submodule.2" not in x and "model.2" not in x]
    encoder_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in encoder_params_list, model.named_parameters()))))
    decoder_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in encoder_params_list, model.named_parameters()))))
    params = [{'params': encoder_params, 'lr': encoder_lr}, {'params': decoder_params, 'lr': decoder_lr}]
    optimizer = torch.optim.Adam(params, lr=decoder_lr)
    return optimizer


def set_lr(optimizer, encoder_lr, decoder_lr):
    optimizer.param_groups[0]['lr'] = encoder_lr
    optimizer.param_groups[1]['lr'] = decoder_lr
    return optimizer


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    UNet, _ = create_unet(device,
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                dropout=0,
                kernel_size=3)
    optimizer = init_lr(UNet, 0, 1e-4)
    print(optimizer.state_dict())
    optimizer = set_lr(optimizer, 1e-4, 0)
    print(optimizer.state_dict())

    for name, param in UNet.named_parameters():
        if param.requires_grad:
            print(name)
