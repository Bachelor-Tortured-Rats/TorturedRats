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
  
  def __init__(self):
    
    # load the paths to the training images 
    self.paths = ['DRIVE/training/images/%02d_training.tif' % i for i in range(21,40)]

  def __len__(self):
      return len(self.paths)

  def __getitem__(self, idx):
    
    # Load image from path and normalize
    img_path = self.paths[idx]
    img = cv2.imread(img_path)#, cv2.IMREAD_GRAYSCALE)
    img = np.array(img)/255
    ### ASBJØRN TESTS
    """
    # Extract center patch and random patch 
    patches, label, patch_locations, _ = extract_patches_and_label_from_center(img, center = (400, 200), patch_size_inner=(80,80), patch_size_outer=(80,80))
    center = patches[0]
    offset_patch = patches[1]

    # Create a figure and 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Plot the arrays in each subplot
    ax1.imshow(img)
    ax2.imshow(offset_patch)
    ax3.imshow(center)
    
    for coordinates in patch_locations.values():
      cv2.rectangle(img, coordinates[0], coordinates[1], (0, 0, 255), 2) # draw rectangle on main image
    
    #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2) # draw rectangle on main image
    #cv2.rectangle(img, (x11, y11), (x22, y22), (0, 0, 255), 2) # draw rectangle on main image
    ax1.imshow(img)


    # Add titles to each subplot
    ax1.set_title('Image')
    ax2.set_title(f'offset_patch_label: {np.argmax(label)}')
    ax3.set_title('center')

    # Add a title to the entire figure
    fig.suptitle('Image with center and random patch')


    # Show the plot
    savepath = 'reports/figures/retinalVessel/' + str(idx)
    plt.savefig(savepath)
    """
    
    # This was our old version
    '''
    # Create all patches
    patches = extract_2D_patches(img, patch_size=(40,40))

    # Provide labels for 
    patch_labels = extract_patch_labels_from_centerpatch(patches, centerPatchCoodinates=(3,3))

    # Make a sample
    center, patch, label = make_sample(patch_labels, smaller_patch_size=(30,30))
    '''
    center_patch, offset_patch, label = generate_patch_pair_MONAI(img, outer_patch_width=50, inner_patch_width=40)
    
    # Convert to torch tensors
    #center = torch.from_numpy(np.expand_dims(center, 0))
    #offset_patch = torch.from_numpy(np.expand_dims(offset_patch, 0))
    label = torch.tensor(label)

    # Create a label
    return center_patch, offset_patch, label

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
    
    dataset.__getitem__(2)
    
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=RetinalVessel_collate_fn)
    
    for i, (patches, labels) in enumerate(dataloader):
        print(f'patches: {patches[0].shape}, {patches[1].shape}')
        print(f'labels: {labels.shape}')

        if i == 1:
            break
    