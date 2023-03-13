from monai.utils import set_determinism

import torch
import wandb
import logging
import click

from src.pipelines.train_model import train_model
from src.models.unet_model import create_unet
from src.data.IRCAD_dataset import load_IRCAD_dataset
from src.data.hepatic_dataset import load_hepatic_dataset


def agent_train_model():
    run = wandb.init()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initializes logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    if wandb.config.get('data_type') == 'IRCAD':
        data_path = '/work3/s204159/3Dircadb1/'
        train_loader, val_loader = load_IRCAD_dataset(data_path, aug=wandb.config.get('augmentation'))
    elif wandb.config.get('data_type') == 'hepatic':
        data_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'
        train_loader, val_loader = load_hepatic_dataset(data_path, aug=wandb.config.get('augmentation'))

    model, params = create_unet(
        device=device,
        spatial_dims=wandb.config.get('spatial_dims'),
        in_channels=wandb.config.get('in_channels'),
        out_channels=wandb.config.get('out_channels'),
        channels=wandb.config.get('channels'),
        strides=wandb.config.get('strides'),
        num_res_units=wandb.config.get('num_res_units'),
        dropout=wandb.config.get('dropout'),
        kernel_size=wandb.config.get('kernel_size')
    )

    train_model(
        model,
        device,
        train_loader,
        val_loader,
        max_epochs=wandb.config.get('epochs'),
        lr=wandb.config.get('lr'),
        data_type=wandb.config.get('data_type'),
        pt="",
        model_save_path='models',
        aug=wandb.config.get('augmentation')
    )

@click.command()
@click.option('--sweep_id', '-id', type=click.STRING, help='The sweep id from the host')
def start_sweep_agent(sweep_id):
    wandb.agent(sweep_id,
                function=agent_train_model,
                count=100,
                project='TorturedRats')

if __name__ == "__main__":
    set_determinism(seed=420)
    start_sweep_agent()