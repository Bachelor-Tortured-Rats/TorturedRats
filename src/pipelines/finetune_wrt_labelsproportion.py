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


def finetune_wrt_labelproportion(data_type, epochs, lr, model_load_path, model_save_path, wandb_logging, augmentation, train_label_proportion, terminate_at_step_count, start_lr, gradlr, setup=''):
    logger = logging.getLogger(__name__)

    # initialize wandb
    config = {
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
        name= f"finetune_{data_type}_s{setup}_lp{train_label_proportion}_tsp{terminate_at_step_count}_id" + str(np.random.randint(10000000)), # makes the name unique
        tags=["finetuning"],
        config=config,
        mode=wandb_logging,
    )

    # load data
    if config['dataset'] == 'IRCAD':
        data_path = '/work3/s204159/3Dircadb1/'
        train_loader, val_loader = load_IRCAD_dataset(data_path, setup=setup, train_label_proportion=train_label_proportion)
    elif config['dataset'] == 'hepatic':
        data_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'
        train_loader, val_loader = load_hepatic_dataset(data_path, setup=config['setup'], train_label_proportion=config['train_label_proportion'])

    # loads model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['setup'] == "3drpl":
        model, params = load_unet_enc(model_load_path, device=device)
    elif config['setup'] == "random":
        model, params = create_unet(device=device)
    else:
        model, params = load_unet(model_load_path, device=device)

    logger.info(f'using model with params: {params}')
    wandb.config.update(params)

    logger.info(
        f'Training model with label proportion {config["train_label_proportion"]}, for {config["train_label_proportion"]} max steps, with learning rate {config["learning_rate"]}')
    model, best_metric, _, _, _, _ = train_model(
        model, device, train_loader, val_loader, max_epochs=epochs, lr=lr, data_type=data_type, pt='finetuned', model_save_path=model_save_path, aug=augmentation, terminate_at_step_count=terminate_at_step_count, start_lr=start_lr, gradlr=gradlr)

    wandb.finish()

    return best_metric

def save_finetune_data(label_proportions, metrics, path):
    # makes sure that the paths exists
    figures_path = os.path.join("./reports/figures", path)
    data_path = os.path.join("./reports/data", path)
    Path(figures_path).mkdir(parents=True, exist_ok=True)
    Path(data_path).mkdir(parents=True, exist_ok=True)

    # creaetes the plots
    fig, ax = plt.subplots()
    ax.plot(label_proportions[:len(
        metrics)], metrics)
    ax.set_xlabel('Label proportion')
    ax.set_ylabel('Best mean dice')
    ax.set_title(f'Best mean dice vs label proportion')
    fig.savefig(os.path.join(figures_path, f'best_mean_dice_vs_label_proportion.png'))

    # saves the data to a text file
    data = np.column_stack((label_proportions, metrics))
    np.savetxt(os.path.join(data_path,
                                f'best_mean_dice_vs_label_proportion.txt'), 
                                data, 
                                delimiter=',', 
                                header="label_proportions, best_mean_dice_list")


@click.command()
@click.option('--data_type', '-d', type=click.Choice(['IRCAD', 'hepatic'], case_sensitive=False), default='IRCAD', help='Dataset choice, defaults to IRCAD')
@click.option('--label_proportions', '-lp', cls=PythonLiteralOption, default=[], help="Which label proportions to use for finetuning")
@click.option('--augmentation', '-a', is_flag=True, help='Toggle using data augmentation')
@click.option('--setup', '-s', type=click.Choice(['transfer', '3drpl', 'random'], case_sensitive=False), default='transfer', help='Which dataset setup to use')
@click.option('--model_load_path', type=click.Path(file_okay=True), help='Path to saved model')
@click.option('--terminate_at_step_count', '-t', type=click.INT, default=None, help="Terminate training after this many steps, defaults to None")
@click.option('--lr', '-lr', type=click.FLOAT, default=1e-4, help='Learning rate, defaults to 1e-4')
@click.option('--saving_path', type=click.Path(), help='The subfolder path to save all the results to')
@click.option('--wandb_logging', '-l', type=click.Choice(['online', 'offline', 'disabled'], case_sensitive=False), default='disabled', help='Should wandb logging be enabled: Can be "online", "offline" or "disabled"')
@click.option('--augmentation', '-a', is_flag=True, help='Toggle using data augmentation')
@click.option('--label_proportions', '-lp', cls=PythonLiteralOption, default=[], help="Which label proportions to use for finetuning")
@click.option('--setup', '-s', type=click.Choice(['transfer', '3drpl', 'random'], case_sensitive=False), default='transfer', help='Which dataset setup to use')
@click.option('--start_lr', '-slr', type=click.FLOAT, default=1e-4, help='Starting learning rate of encoder, defaults to 1e-4')
@click.option('--gradlr', '-g', is_flag=True, help='Toggle gradually increasing encoder lr (only applicable when start_lr is set)')
def main(data_type, epochs, lr, model_load_path, model_save_path, figures_save_path, wandb_logging, augmentation, label_proportions, terminate_at_step_count, setup, start_lr, gradlr):
    # initializes logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Initialized logging')
    logger.info('Running finetune_wrt_labelproportion.py')
    logger.info(f'Using dataset {data_type}')
    logger.info(f'Using setup {setup}')

    # trains the model for each label proportion
    best_metric_list = []
    for label_proportion in label_proportions:
        set_determinism(seed=420)

        label_proportion = label_proportion
        best_metric = finetune_wrt_labelproportion(
            data_type, None, lr, model_load_path, saving_path, wandb_logging, augmentation, label_proportion, terminate_at_step_count, setup=setup, start_lr=start_lr, gradlr=gradlr)

        best_metric_list.append(best_metric)
        logger.info(
            f'Best metric for label proportion {label_proportion}: {best_metric}')
        
        save_finetune_data(label_proportion, best_metric, saving_path)

    # prints the final data
    logger.info(f'FINAL DATA:  {label_proportions}: {best_metric_list}')


if __name__ == "__main__":
    main()
