import logging
import pdb
from matplotlib import pyplot as plt

import numpy as np
import torch
from monai.data import DataLoader
from monai.networks.layers import Norm
from torch import nn
from torch.utils.data import DataLoader

from src.models.unet_enc_model import create_unet_enc
from src.self_supervised_MVP.model import (Beefier_Pred_head, BeefierEncoder,
                                           SelfSupervisedModel)
from src.self_supervised_MVP.retinalVesselDataset import (
    RetinalVessel_collate_fn, RetinalVesselDataset)


def train_model(model, device, train_loader, val_loader, max_epochs, lr):
    logger = logging.getLogger(__name__)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    epoch_loss_values = []
    val_interval = 2
    metric_values = []
    val_loss_list = []
    best_metric = 9999999
    best_metric_epoch = -1

    for epoch in range(max_epochs):
        logger.info("-" * 10)
        logger.info(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        # training part
        for ((center_patch, offset_patch), labels) in train_loader:
            step += 1
            
            center_patch, offset_patch, labels = center_patch.to(device), offset_patch.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(center_patch, offset_patch)
            loss = loss_function(outputs, labels)
            
            loss.backward()
            
            optimizer.step()
            epoch_loss += loss.item()

        # calculates average loss in epoch
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # validates every val_interval epochs
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():

                val_loss = 0
                classified_correct = []
                for ((center_patch, offset_patch), labels) in val_loader:
                    center_patch, offset_patch, labels = center_patch.to(device), offset_patch.to(device), labels.to(device)

                    # forward pass on validation data
                    outputs = model(center_patch, offset_patch)

                    # calculates validation loss
                    loss = loss_function(outputs, labels)
                    val_loss += loss.item()
                    
                    classified_correct += outputs.argmax(dim=1).cpu()==labels.cpu()

                # saves validation loss
                metric_values.append(np.mean(classified_correct))
                val_loss_list.append(val_loss)

                # updates if the current metric is better than the best metric
                if val_loss < best_metric:
                    best_metric = val_loss
                    best_metric_epoch = epoch + 1
                    
                logger.info(
                    f"current val loss: {val_loss:.4f} and accuracy {metric_values[-1]:.4f} at epoch {epoch + 1} \n best val loss: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )

    plt.figure("train", (16, 6))
    plt.subplot(1, 3, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 3, 2)
    plt.title("val Average Loss")
    x = [val_interval * (i + 1) for i in range(len(val_loss_list))]
    y = val_loss_list
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 3, 3)
    plt.title("Val acc")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig(f'reports/figures/training_graph_{max_epochs}_{lr}.png')

    return None

if __name__ == "__main__":
    # Set the device to use for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # creates models
    encoder = BeefierEncoder()
    uNetEnc, _ = create_unet_enc(device,
                                 spatial_dims=2,
                                 in_channels=3,
                                 channels=(16, 32, 64, 128),
                                 strides=(2, 2, 2),)
    prediction_head = Beefier_Pred_head()
    selfSupervisedModel = SelfSupervisedModel(uNetEnc, prediction_head)
    selfSupervisedModel.to(device)

    # create dataloaders
    dataset_train = RetinalVesselDataset(train_data=True)
    dataset_val = RetinalVesselDataset(train_data=False)
    train_loader = DataLoader(dataset_train, batch_size=4, shuffle=True, collate_fn=RetinalVessel_collate_fn)
    val_loader = DataLoader(dataset_val, batch_size=4, shuffle=False, collate_fn=RetinalVessel_collate_fn)

    # trains models
    train_model(selfSupervisedModel, device, train_loader,val_loader, max_epochs=500, lr=1e-3)