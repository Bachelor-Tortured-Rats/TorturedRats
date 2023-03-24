import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Encoder, Pred_head, Beefier_Pred_head, BeefierEncoder

import torch
import torch.nn.functional as F

from src.self_supervised_MVP.retinalVesselDataset import RetinalVesselDataset, RetinalVessel_collate_fn

def train_loop(encoder, prediction_head, train_loader, optimizer, device):
    # Set the model to training mode
    encoder.train()
    prediction_head.train()
    
    # Number of correct predictions
    correct = 0

    # Number of predictions
    total = 0

    # loss stats
    total_loss = 0
    steps = 0

    # Loop over the training data
    for data, labels in train_loader:
        steps += 1
        
        data_patch_center, data_patch_offset = data
        # Move the data and labels to the device
        data_patch_center = data_patch_center.to(device)
        data_patch_offset = data_patch_offset.to(device)
        labels = labels.to(device)

        # Ensure that all data is of type float32
        data_patch_center = data_patch_center.float()
        data_patch_offset = data_patch_offset.float()
        labels = labels.float()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass through the encoder
        latent_patch_center = encoder(data_patch_center)
        latent_patch_offset = encoder(data_patch_offset)
        
        # Forward pass through the prediction head
        prediction = prediction_head(latent_patch_center, latent_patch_offset)

        # Compute the loss based on the relative location of the patches
        loss = F.cross_entropy(prediction.squeeze(), labels.long())
        total_loss += loss.item()


        # Compute the number of correct predictions
        _, predicted = torch.max(prediction.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        
        # Save the encoder state dict
        encoder_state_dict = encoder.state_dict()
        torch.save(encoder_state_dict, 'encoder_state_dict.pth')
        
    # Return the average loss over the training data
    return correct / total, total_loss / steps


if __name__ == "__main__":
    import numpy as np
    #encoder = Encoder()
    #prediction_head = Pred_head()
    encoder = BeefierEncoder()
    prediction_head = Beefier_Pred_head()

    dataset = RetinalVesselDataset()

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=RetinalVessel_collate_fn)

    
    # Set the device to use for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    # Set the models and optimizer to use the device
    encoder.to(device)
    prediction_head.to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(prediction_head.parameters()), lr=0.00001)

    # Train the models using the custom loss function and the train loop
    longrun_avg = list(np.zeros(100))
    for epoch in range(10000):
        train_loss, loss = train_loop(encoder, prediction_head, train_loader, optimizer, device)
        longrun_avg.append(train_loss)
        longrun_avg.pop(0)
        print(f"Epoch {epoch+1}, Accuracy: {train_loss:.4f}, Avg Accuracy: {np.mean(longrun_avg):.4f}")
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
        

    