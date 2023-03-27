import os
import pandas as pd
from torchvision.io import read_image
import cv2
import numpy as np
from PatchMaker_2D import *
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

class RetinalVesselDataset(Dataset):
  
  def __init__(self, train_data=True):
    
    # load the paths to the training images 
    if train_data:
      self.paths = [f'DRIVE/training/images/{i:02d}_training.tif' for i in range(21,40)]
    else:
      self.paths = [f'DRIVE/test/images/{i:02d}_test.tif' for i in range(1,21)]

  def __len__(self):
      return len(self.paths)

  def __getitem__(self, idx):
    
    # Load image from path and normalize
    img_path = self.paths[idx]
    img = cv2.imread(img_path)#, cv2.IMREAD_GRAYSCALE)
    img = np.array(img)/255

    center_patch, offset_patch, label = generate_patch_pair_MONAI(img, outer_patch_width=50, inner_patch_width=40, num_pairs=10)
    label = torch.tensor(label)

    return center_patch, offset_patch, label

def RetinalVessel_collate_fn(batch):
  """
  A custom collate function for the RetinalVesselDataset
  """

  # Unzip the batch
  #center_patches = [sample[0] for sample in batch]
  center_patches = [patch for sample in batch for patch in sample[0]]
  #offset_patches = [sample[1] for sample in batch]
  offset_patches = [patch for sample in batch for patch in sample[1]]
  #labels = [sample[2] for sample in batch]
  labels = [patch for sample in batch for patch in sample[2]]

  # Stack the patches and labels
  center_patches = torch.stack(center_patches, dim=0)
  offset_patches = torch.stack(offset_patches, dim=0)
  labels = torch.stack(labels, dim=0)

  return (center_patches.float(), offset_patches.float()), labels
    
if __name__ == "__main__":
    
    from torch.utils.data import DataLoader

    dataset = RetinalVesselDataset()
    
    dataset.__getitem__(2)
    
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=RetinalVessel_collate_fn)
    
    for i, (patches, labels) in enumerate(dataloader):
        print(f'patches: {patches[0].shape}, {patches[1].shape}')
        print(f'labels: {labels.shape}')

        if i == 1:
            break
    