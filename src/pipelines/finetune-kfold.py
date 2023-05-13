import argparse
import logging
import os

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from monai.utils import set_determinism

import wandb
from src.data.hepatic_dataset import load_hepatic_dataset
from src.data.IRCAD_dataset import load_IRCAD_dataset
from src.models.unet_enc_model import init_lr, load_unet_enc, set_lr
from src.models.unet_model import create_unet, load_unet


def train_model(model, terminate_at_step, eval_each_steps, train_loader, val_loader, encoder_lr, learning_rate, increase_encoder_lr, device):
    '''
        Trains model for terminate_at_step steps, and evaluates model every eval_each_steps steps.
    '''

    logger = logging.getLogger(__name__)

    step = 0
    best_dice_metric = -1
    best_dice_metric_step = -1
    train_iteration_loss_values = []
    dice_metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    optimizer = init_lr(model, encoder_lr=encoder_lr, decoder_lr=learning_rate)

    # we terminate training at terminate_at_step
    while True:
        model.train()
        train_iteration_loss = 0

        # trains model for eval_each_steps steps
        while step % eval_each_steps != 0:
            for batch_data in train_loader:
                step += 1

                # stops in middle of epoch for evaluation
                if eval_each_steps % step != 0:
                    break
                # stops when reaching terminate_at_step
                if step == terminate_at_step:
                    return
                # increases encoder lr
                if increase_encoder_lr and step % int(terminate_at_step/10) == 0:
                    set_lr(optimizer, encoder_lr+(learning_rate-encoder_lr)
                           * (step/terminate_at_step), learning_rate)
                    logger.info(
                        f"encoder learning rate increased to {optimizer.param_groups[0]['lr']:.5f}")

                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                train_iteration_loss += loss.item()

        train_iteration_loss /= eval_each_steps
        train_iteration_loss_values.append(train_iteration_loss)

        # we evaluate the model every val_interval epochs
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i)
                               for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i)
                              for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            dice_metric_value = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            dice_metric_values.append(dice_metric_value)
            if dice_metric_value > best_dice_metric:
                logger.info(
                    "New best model found, with dice metric: {dice_metric_value:.4f} at step {step}")
                best_dice_metric = dice_metric_value
                best_dice_metric_step = step

            logger.info(
                f"step: {step} of {terminate_at_step}, dice_metric: {dice_metric_value:.4f}")
            
            wandb.log(step=step,data={
                "dice_metric_values": dice_metric_values,
                "best_dice_metric" : best_dice_metric,
                "best_dice_metric_step": best_dice_metric_step,
                "train_iteration_loss": train_iteration_loss,
                "encoder_learning_rate": optimizer.param_groups[0]['lr'],
            })


def main(jobid: str, data_type, k_fold, label_proportion, model_load_path, setup, terminate_at_step, eval_each_steps, encoder_lr, learning_rate, increase_encoder_lr, experiement_name, wandb_logging):
    set_determinism(seed=420)
    logger = logging.getLogger(__name__)

    # initialize wandb
    config = {
        'jobid': jobid,
        'data_type': data_type,
        'label_proportion': label_proportion,
        'model_load_path': model_load_path,
        'setup': setup,
        'terminate_at_step': terminate_at_step,
        'setup': setup,
        'encoder_lr': encoder_lr,
        'learning_rate': learning_rate,
        'increase_encoder_lr': increase_encoder_lr,
    }

    # creates params for wandb
    init_params = {
        'project': "TorturedRats",
        'entity': "team-christian",
        'config': config,
        'mode': wandb_logging,
        'group': f"{experiement_name}_lp_{k_fold}",
    }

    # if running on cluster, add job id to name
    if jobid != 00000000:
        init_params['name'] = jobid

    run = wandb.init(**init_params)

    # load data
    if config['data_type'] == 'IRCAD':
        data_path = '/work3/s204159/3Dircadb1/'
        train_loader, val_loader = load_IRCAD_dataset(
            data_path, setup=setup, train_label_proportion=config['label_proportion'],k_fold=k_fold)
    elif config['data_type'] == 'hepatic':
        data_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'
        train_loader, val_loader = load_hepatic_dataset(
            data_path, k_fold, setup=config['setup'], train_label_proportion=config['label_proportion'])

    # loads model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['setup'] == "3drpl":
        model, params = load_unet_enc(model_load_path, device=device)
    elif config['setup'] == "random":
        model, params = create_unet(device=device)
    elif config['setup'] == "transfer":
        model, params = load_unet(model_load_path, device=device)

    logger.info(f'using model with params: {params}')
    wandb.config.update(params)

    # train model
    train_model(model, terminate_at_step, eval_each_steps, train_loader, val_loader, encoder_lr, learning_rate, increase_encoder_lr, device)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--jobid', type=int, default=00000000,
                        help='Job id, defaults to 00000')
    parser.add_argument('--data_type', '-d', choices=['IRCAD', 'hepatic'], default='IRCAD',
                        help='Dataset choice, defaults to IRCAD')
    parser.add_argument('--label_proportion', '-lp', type=float,
                        help="Which label proportion to use for finetuning")
    parser.add_argument('--model_load_path', type=str,
                        help='Path to saved model')
    parser.add_argument('--setup', '-s', choices=['transfer', '3drpl', 'random'],
                        help='Which dataset setup to use')
    parser.add_argument('--terminate_at_step', '-t', type=int, default=24000,
                        help="Terminate training after this many steps, defaults to 24000")
    parser.add_argument('--eval_each_steps', type=int, default=200,
                        help="Evaluate model each n steps, defaults to 200")
    parser.add_argument('--k_fold', type=int, default=1,
                        help="kfold to use, defaults to 1")
    parser.add_argument('--encoder_lr', type=float, default=1e-4,
                        help='Starting learning rate of encoder, defaults to 1e-4')
    parser.add_argument('--increase_encoder_lr', action='store_true',
                        help='Toggle gradually increasing encoder lr (only applicable when start_lr is set)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate for rest of model, defaults to 1e-4')
    parser.add_argument('--wandb_logging', '-l', choices=['online', 'offline', 'disabled'],
                        help='Should wandb logging be enabled: Can be "online", "offline" or "disabled"')
    parser.add_argument('--experiement_name', type=str,
                        help='Wandb experiement name, should be unique for each experiment')

    args = parser.parse_args() 

    
    main(str(args.jobid), args.data_type, args.k_fold, args.label_proportion, args.model_load_path, args.setup, args.terminate_at_step, args.eval_each_steps, args.encoder_lr, args.learning_rate, args.increase_encoder_lr, args.experiement_name, args.wandb_logging)