import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.utils import set_determinism

import wandb
from src.data.hepatic_dataset import load_hepatic_dataset
from src.data.IRCAD_dataset import load_IRCAD_dataset
from src.models.unet_enc_model import load_unet_enc
from src.models.unet_model import create_unet, load_unet
from src.pipelines.train_model import train_model
from src.utils.click_utils import PythonLiteralOption


def finetune_wrt_labelproportion(data_type, epochs, lr, model_load_path, model_save_path, wandb_logging, augmentation, train_label_proportion, terminate_at_step_count, setup=''):
    logger = logging.getLogger(__name__)

    # initialize wandb
    config = {
        "epochs": epochs,
        "learning_rate": lr,
        "pretrained": True,
        "dataset": data_type,
        "augmentation": augmentation,
        "train_label_proportion": train_label_proportion,
        "terminate_at_step_count": terminate_at_step_count,
        "setup": setup,
    }

    run = wandb.init(
        project="TorturedRats",
        name= f"{data_type}_tsp{terminate_at_step_count}__aug{augmentation}_lr{lr:.1E}_s{setup}_lp{train_label_proportion}",
        tags=["finetuning"],
        config=config,
        mode=wandb_logging,
    )

    # load data
    if data_type == 'IRCAD':
        data_path = '/work3/s204159/3Dircadb1/'
        train_loader, val_loader = load_IRCAD_dataset(data_path, setup=setup, train_label_proportion=train_label_proportion)
    elif data_type == 'hepatic':
        data_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'
        train_loader, val_loader = load_hepatic_dataset(data_path, setup=setup, train_label_proportion=train_label_proportion)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if setup == "3drpl":
        model, params = load_unet_enc(model_load_path, device=device)
    elif setup == "random":
        model, params = create_unet(device=device)
    else:
        model, params = load_unet(model_load_path, device=device)

    logger.info(f'using model with params: {params}')
    wandb.config.update(params)

    logger.info(
        f'Training model with label proportion {train_label_proportion}, for {epochs} epochs, with learning rate {lr}')
    model, best_metric, _, _, _, _ = train_model(
        model, device, train_loader, val_loader, max_epochs=epochs, lr=lr, data_type=data_type, pt='finetuned', model_save_path=model_save_path, aug=augmentation, terminate_at_step_count=terminate_at_step_count)

    wandb.finish()

    return best_metric


@click.command()
@click.option('--data_type', '-d', type=click.Choice(['IRCAD', 'hepatic'], case_sensitive=False), default='IRCAD', help='Dataset choice, defaults to IRCAD')
@click.option('--epochs', '-e', type=click.INT, default=20, help='Max epochs to train for, defaults to 20')
@click.option('--terminate_at_step_count', '-t', type=click.INT, default=None, help="Terminate training after this many steps, defaults to None")
@click.option('--lr', '-lr', type=click.FLOAT, default=1e-4, help='Learning rate, defaults to 1e-4')
@click.option('--model_load_path', type=click.Path(file_okay=True), help='Path to saved model')
@click.option('--model_save_path', type=click.Path(), default='models', help='Path to folder for saving model')
@click.option('--figures_save_path', type=click.Path(), default='reports/figures/finetune_wrt_labelproportion', help='Path to folder for saving figures')
@click.option('--wandb_logging', '-l', type=click.Choice(['online', 'offline', 'disabled'], case_sensitive=False), default='disabled', help='Should wandb logging be enabled: Can be "online", "offline" or "disabled"')
@click.option('--augmentation', '-a', is_flag=True, help='Toggle using data augmentation')
@click.option('--label_proportions', '-lp', cls=PythonLiteralOption, default=[], help="Which label proportions to use for finetuning")
@click.option('--setup', '-s', type=click.Choice(['transfer', '3drpl', 'random'], case_sensitive=False), default='transfer', help='Which dataset setup to use')
def main(data_type, epochs, lr, model_load_path, model_save_path, figures_save_path, wandb_logging, augmentation, label_proportions, terminate_at_step_count, setup):
    # initializes logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Initialized logging')
    logger.info(f'Using dataset {data_type}')
    logger.info(f'Using setup {setup}')

    # makes sure that the figures_save_path folder path exists
    Path(figures_save_path).mkdir(parents=True, exist_ok=True)

    best_mean_dice_list = []
    for label_proportion in label_proportions:
        label_proportion = float(label_proportion)

        set_determinism(seed=420)
        best_mean_dice = finetune_wrt_labelproportion(
            data_type, epochs, lr, model_load_path, model_save_path, wandb_logging, augmentation, label_proportion, terminate_at_step_count, setup=setup)

        best_mean_dice_list.append(best_mean_dice)
        logger.info(
            f'Best mean dice for label proportion {label_proportion}: {best_mean_dice}')

        # code to plot the best mean dice vs label proportion saves to figures_save_path
        # updates after each label proportion
        fig, ax = plt.subplots()
        ax.plot(label_proportions[:len(
            best_mean_dice_list)], best_mean_dice_list)
        ax.set_xlabel('Label proportion')
        ax.set_ylabel('Best mean dice')
        ax.set_title(
            f'Best mean dice vs label proportion, total training steps: {terminate_at_step_count}')
        fig.savefig(os.path.join(figures_save_path,
                                 f'{setup}_{terminate_at_step_count}_best_mean_dice_vs_label_proportion_{label_proportions}.png'))

        # saves the data to a text file
        data = np.array(
            [label_proportions[:len(best_mean_dice_list)], best_mean_dice_list])
        np.savetxt(os.path.join(figures_save_path,
                                f'{setup}_{terminate_at_step_count}_best_mean_dice_vs_label_proportion_{label_proportions}.txt'), data, delimiter=',', header="label_proportions, best_mean_dice_list")

    logger.info(f'FINAL DATA:  {label_proportions}: {best_mean_dice_list}')


if __name__ == "__main__":
    print('Running finetune_wrt_labelproportion.py')

    main()
