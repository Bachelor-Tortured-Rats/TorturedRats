from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
import torch
import matplotlib.pyplot as plt
import click
import wandb
import logging


from src.models.unet_model import create_unet, load_unet
from src.data.IRCAD_dataset import load_IRCAD_dataset
from src.data.hepatic_dataset import load_hepatic_dataset


def train_model(model, device, train_loader, val_loader, max_epochs, lr, data_type, pt, model_save_path, aug):
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # logging setup
    wandb.watch(model, criterion=loss_function, log="all", log_freq=2)
    logger = logging.getLogger(__name__)

    max_epochs = max_epochs
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(max_epochs):
        logger.info("-" * 10)
        logger.info(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        # logger.info("before train_loader")
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            # logger.info("before optimizer.zero_grad")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            # logger.info("before loss.backward()")
            loss.backward()
            # logger.info("before optimizer.step()")
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        if (epoch + 1) % val_interval == 0:
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
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'spatial_dims': model.dimensions,
                        'in_channels': model.in_channels,
                        'out_channels': model.out_channels,
                        'channels': model.channels,
                        'strides': model.strides,
                        'num_res_units': model.num_res_units,
                        'dropout': model.dropout,
                        'kernel_size': model.kernel_size,
                        'epoch': best_metric_epoch,
                        'best_metric': best_metric,
                    }, "{folder_path}/{data_type}_{pt}_e{max_epochs}_k{kernel_size}_d{dropout}_lr{lr:.0E}_a{aug}_bmm.pth".format(
                        folder_path=model_save_path,
                        data_type=data_type,
                        pt=pt,
                        max_epochs=max_epochs,
                        lr=lr,
                        aug=aug,
                        kernel_size=model.kernel_size,
                        dropout=model.dropout
                    )
                    )
                    logger.info("saved new best metric model")

                wandb.log(step=epoch, data={
                          "mean_dice": metric, "best_mean_dice": best_metric, "train_loss": epoch_loss})
                logger.info(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
    return model, best_metric, best_metric_epoch, epoch_loss_values, val_interval, metric_values


def display_model_training(best_metric, best_metric_epoch, epoch_loss_values, val_interval, metric_values, figures_save_path):
    print(
        f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig(f'{figures_save_path}/training_graph.png')


@click.command()
@click.option('--data_type', '-d', type=click.Choice(['IRCAD', 'hepatic'], case_sensitive=False), default='IRCAD', help='Dataset choice, defaults to IRCAD')
@click.option('--pretrained', '-p', type=click.Path(exists=False), default='', help='Path to existing model if finetuning, defaults to none')
@click.option('--epochs', '-e', type=click.INT, default=20, help='Max epochs to train for, defaults to 20')
@click.option('--lr', '-lr', type=click.FLOAT, default=1e-4, help='Learning rate, defaults to 1e-4')
@click.option('--model_save_path', type=click.Path(exists=True), default='models', help='Path to folder for saving model')
@click.option('--figures_save_path', type=click.Path(exists=True), default='reports/figures/train_model', help='Path to folder for saving figures')
@click.option('--wandb_logging', '-l', type=click.Choice(['online', 'offline', 'disabled'], case_sensitive=False), default='disabled', help='Should wandb logging be enabled: Can be "online", "offline" or "disabled"')
@click.option('--augmentation', '-a', is_flag=True, help='Toggle using data augmentation')
@click.option('--kernel_size', '-k', type=click.INT, default=3, help='Kernel size')
@click.option('--dropout', '-dr', type=click.FLOAT, default=0, help='Dropout')
def main(data_type, pretrained, epochs, lr, model_save_path, figures_save_path, wandb_logging, augmentation, kernel_size, dropout):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initializes logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Initialized logging')
    logger.info(f'Using dataset {data_type}')

    # initializing wandb
    config = {
        "epochs": epochs,
        "learning_rate": lr,
        "pretrained": pretrained,
        "dataset": data_type,
        "augmentation": augmentation,
        "kernel_size": kernel_size,
        "dropout": dropout
    }

    run = wandb.init(
        project="TorturedRats",
        notes="test run",
        tags=["testing"],
        config=config,
        mode=wandb_logging,
    )

    if data_type == 'IRCAD':
        data_path = '/work3/s204159/3Dircadb1/'
        train_loader, val_loader = load_IRCAD_dataset(data_path, aug=augmentation)
    elif data_type == 'hepatic':
        data_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'
        train_loader, val_loader = load_hepatic_dataset(data_path, aug=augmentation)

    if pretrained != '':
        model, params = load_unet(pretrained, device=device)
        pt = 'finetuned'
    else:
        model, params = create_unet(device=device, kernel_size=kernel_size, dropout=dropout)
        pt = 'standard'

    if wandb_logging:
        wandb.config.update(params)
    else:
        logger.info(f'using model with params: {params}')

    model, best_metric, best_metric_epoch, epoch_loss_values, val_interval, metric_values = train_model(
        model, device, train_loader, val_loader, max_epochs=epochs, lr=lr, data_type=data_type, pt=pt, model_save_path=model_save_path, aug=augmentation)

    display_model_training(best_metric, best_metric_epoch, epoch_loss_values,
                           val_interval, metric_values, figures_save_path)

    wandb.finish()


if __name__ == "__main__":
    set_determinism(seed=420)

    main()
