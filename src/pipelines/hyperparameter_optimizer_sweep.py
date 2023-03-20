import wandb
import logging

from monai.utils import set_determinism

def start_sweep():
    """Starts a sweep host with wandb, use "wandb agent sweep_id" to start an agent.
    """
    # initializes logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    # Define sweep config
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep_aug_hepatic',
        'metric': {'goal': 'maximize', 'name': 'best_mean_dice'},
        'parameters':
        {
            'spatial_dims' : {'value': 3},
            'in_channels' : {'value': 1},
            'out_channels' : {'value': 2},
            'channels' : {'value': (16, 32, 64, 128, 256)},
            'strides' : {'value': (2, 2, 2, 2)},
            'num_res_units' : {'values': [0,2,4]},
            'epochs': {'value': 100},
            'lr': {'values': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]},
            # 'lr': {'max': 1e-2, 'min': 1e-5},
            'data_type': {'value': 'hepatic'},
            'augmentation': {'value': True},
            'dropout': {'values': [0.0, 0.1, 0.2, 0.4]},
            'kernel_size': {'values': [3, 5]}
        }
    }

    # Initialize sweep by passing in config. (Optional) Provide a name of the project.
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='TorturedRats',entity='team-christian')

    logging.warning(f'Remember to close sweep with id {sweep_id} after use on wandb.ai')
    wandb.finish()


if __name__ == "__main__":
    set_determinism(seed=420)

    start_sweep()
