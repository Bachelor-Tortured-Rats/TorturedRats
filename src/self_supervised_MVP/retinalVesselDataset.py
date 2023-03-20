import os
import pandas as pd
from torchvision.io import read_image
import cv2
import numpy as np
from PatchMaker_2D import *
from torch.utils.data import Dataset
import torch

class RetinalVesselDataset(Dataset):
  
  def __init__(self):
    
    # load the paths to the training images 
    self.paths = ['DRIVE/training/images/%02d_training.tif' % i for i in range(21,40)]

  def __len__(self):
      return len(self.paths)

  def __getitem__(self, idx):
    
    # Load image from path and normalize
    img_path = self.paths[idx]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.array(img)/255
    
    # Create all patches
    patches = extract_2D_patches(img, patch_size=(40,40))

    # Provide labels for 
    patch_labels = extract_patch_labels_from_centerpatch(patches, centerPatchCoodinates=(3,3))

    # Make a sample
    center, patch, label = make_sample(patch_labels, smaller_patch_size=(30,30))
    
    # Convert to torch tensors
    center = torch.from_numpy(np.expand_dims(center, 0))
    patch = torch.from_numpy(np.expand_dims(patch, 0))
    label = torch.from_numpy(label)

    # Create a label
    return center, patch, label

def RetinalVessel_collate_fn(batch):
  """
  A custom collate function for the RetinalVesselDataset
  """

  # Unzip the batch
  center_patches = [sample[0] for sample in batch]
  offset_patches = [sample[1] for sample in batch]
  labels = [sample[2] for sample in batch]

  # Stack the patches and labels
  center_patches = torch.stack(center_patches, dim=0)
  offset_patches = torch.stack(offset_patches, dim=0)
  labels = torch.stack(labels, dim=0)
      
  return (center_patches, offset_patches), labels
    
if __name__ == "__main__":
    
    from torch.utils.data import DataLoader

    dataset = RetinalVesselDataset()
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=RetinalVessel_collate_fn)
    
    for i, (patches, labels) in enumerate(dataloader):
        print(f'patches: {patches[0].shape}, {patches[1].shape}')
        print(f'labels: {labels.shape}')

        if i == 1:
            break