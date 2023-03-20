import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.self_supervised.model import Encoder, Pred_head

import torch
import torch.nn.functional as F

def train_loop(encoder, prediction_head, train_loader, optimizer, device):
    # Set the model to training mode
    encoder.train()
    prediction_head.train()
    
    # Loop over the training data
    for data, labels in train_loader:
        
        data_patch_center, data_patch_offset = data
        # Move the data and labels to the device
        data_patch_center = data_patch_center.to(device)
        data_patch_offset = data_patch_offset.to(device)
        labels = labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass through the encoder
        latent_patch_center = encoder(data_patch_center)
        latent_patch_offset = encoder(data_patch_offset)
        
        # Forward pass through the prediction head
        prediction = prediction_head(latent_patch_center, latent_patch_offset)
        
        # Compute the loss based on the relative location of the patches
        loss = F.binary_cross_entropy_with_logits(prediction.squeeze(), labels)
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        
    # Return the average loss over the training data
    return loss.item()


if __name__ == "__main__":

    # Load the CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Set the device to use for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the models and optimizer to use the device
    encoder.to(device)
    prediction_head.to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(prediction_head.parameters()), lr=0.001)

    # Train the models using the custom loss function and the train loop
    for epoch in range(10):
        train_loss = train_loop(encoder, prediction_head, train_loader, optimizer, device, loss_fn=patch_location_loss)
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")