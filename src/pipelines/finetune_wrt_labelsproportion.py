import logging
import os

import click
import torch
from matplotlib import pyplot as plt
from monai.utils import set_determinism

import wandb
from src.data.hepatic_dataset import load_hepatic_dataset
from src.data.IRCAD_dataset import load_IRCAD_dataset
from src.models.unet_model import create_unet, load_unet
from src.pipelines.train_model import train_model
from src.utils.click_utils import PythonLiteralOption


def finetune_wrt_labelproportion(data_type, epochs, lr, model_load_path, model_save_path, wandb_logging, augmentation, train_label_proportion):
    logger = logging.getLogger(__name__)

    # initialize wandb
    config = {
        "epochs": epochs,
        "learning_rate": lr,
        "pretrained": True,
        "dataset": data_type,
        "augmentation": augmentation,
        "train_label_proportion": train_label_proportion,
    }

    run = wandb.init(
        project="TorturedRats",
        # notes="",
        tags=["finetuning", "data_type"],
        config=config,
        mode=wandb_logging,
    )

    # load data
    if data_type == 'IRCAD':
        data_path = '/work3/s204159/3Dircadb1/'
        train_loader, val_loader = load_IRCAD_dataset(data_path,
            aug=augmentation, train_label_proportion=train_label_proportion)
    elif data_type == 'hepatic':
        data_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'
        train_loader, val_loader = load_hepatic_dataset(data_path,
            aug=augmentation, train_label_proportion=train_label_proportion)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, params = load_unet(model_load_path, device=device)
    logger.info(f'using model with params: {params}')
    wandb.config.update(params)

    logger.info(f'Training model with label proportion {train_label_proportion}, for {epochs} epochs, with learning rate {lr}')
    model, best_metric, _, _, _, _ = train_model(
        model, device, train_loader, val_loader, max_epochs=epochs, lr=lr, data_type=data_type, pt='finetuned', model_save_path=model_save_path, aug=augmentation)

    return best_metric

@click.command()
@click.option('--data_type', '-d', type=click.Choice(['IRCAD', 'hepatic'], case_sensitive=False), default='IRCAD', help='Dataset choice, defaults to IRCAD')
@click.option('--epochs', '-e', type=click.INT, default=20, help='Max epochs to train for, defaults to 20')
@click.option('--lr', '-lr', type=click.FLOAT, default=1e-4, help='Learning rate, defaults to 1e-4')
@click.option('--model_load_path', type=click.Path(file_okay=True), help='Path to saved model')
@click.option('--model_save_path', type=click.Path(exists=True), default='models', help='Path to folder for saving model')
@click.option('--figures_save_path', type=click.Path(), default='reports/figures/finetune_wrt_labelproportion', help='Path to folder for saving figures')
@click.option('--wandb_logging', '-l', type=click.Choice(['online', 'offline', 'disabled'], case_sensitive=False), default='disabled', help='Should wandb logging be enabled: Can be "online", "offline" or "disabled"')
@click.option('--augmentation', '-a', is_flag=True, help='Toggle using data augmentation')
@click.option('--label_proportions','-lp', cls=PythonLiteralOption, default=[],help="Which label proportions to use for finetuning")
def main(data_type, epochs, lr, model_load_path, model_save_path, figures_save_path, wandb_logging, augmentation, label_proportions):
    # initializes logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Initialized logging')
    logger.info(f'Using dataset {data_type}')

    best_mean_dice_list = []
    for label_proportion in label_proportions:
        label_proportion = float(label_proportion)
        best_mean_dice = finetune_wrt_labelproportion(
            data_type, epochs, lr, model_load_path, model_save_path, wandb_logging, augmentation, label_proportion)
        best_mean_dice_list.append(best_mean_dice)
        logger.info(f'Best mean dice for label proportion {label_proportion}: {best_mean_dice}')

    # code to plot the best mean dice vs label proportion saves to figures_save_path
    fig, ax = plt.subplots()
    ax.plot(label_proportions, best_mean_dice_list)
    ax.set_xlabel('Label proportion')
    ax.set_ylabel('Best mean dice')
    ax.set_title('Best mean dice vs label proportion')
    fig.savefig(os.path.join(figures_save_path,
                'best_mean_dice_vs_label_proportion.png'))


if __name__ == "__main__":
    print('Running finetune_wrt_labelproportion.py')
    set_determinism(seed=420)

    main()

