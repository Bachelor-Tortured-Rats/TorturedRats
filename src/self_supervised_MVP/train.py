from monai.data import DataLoader
import torch
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from monai.networks.layers import Norm

from model import Beefier_Pred_head, BeefierEncoder, SelfSupervisedModel
from src.utils.models import UNetEnc

import torch
import pdb

from src.self_supervised_MVP.retinalVesselDataset import RetinalVesselDataset, RetinalVessel_collate_fn

def train_model(model, device, train_loader, val_loader, max_epochs, lr):
    logger = logging.getLogger(__name__)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    epoch_loss_values = []
    val_interval = 2
    metric_values = []
    best_metric = -1
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
                for ((center_patch, offset_patch), labels) in val_loader:
                    center_patch, offset_patch, labels = center_patch.to(device), offset_patch.to(device), labels.to(device)

                    # forward pass on validation data
                    outputs = model(center_patch, offset_patch)

                    # calculates validation loss
                    loss = loss_function(outputs, labels)
                    val_loss += loss.item()

                # saves validation loss
                metric_values.append(val_loss)
                
                # updates if the current metric is better than the best metric
                if val_loss > best_metric:
                    best_metric = val_loss
                    best_metric_epoch = epoch + 1
                    
                logger.info(
                    f"current epoch: {epoch + 1} current mean dice: {val_loss:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

    return None

if __name__ == "__main__":
    # Set the device to use for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = BeefierEncoder()
    uNetEnc = UNetEnc(spatial_dims=3,
                in_channels=1,
                out_channels=2,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                dropout=0,
                kernel_size=3,
                norm=Norm.BATCH,)
    prediction_head = Beefier_Pred_head()
    selfSupervisedModel = SelfSupervisedModel(encoder, prediction_head)
    selfSupervisedModel.to(device)

    dataset = RetinalVesselDataset()
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=RetinalVessel_collate_fn)

    train_model(selfSupervisedModel, device, train_loader,train_loader, max_epochs=10, lr=0.0001)