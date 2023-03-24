import os
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

import random


def outcrop_random_patch(image, size = 25):
    """ Helper function to outcrop patch """

    # Get image shape
    img_arr = np.array(image) 
    h, w = img_arr.shape 

    # Select random patch location and extract patch
    x = random.randint(0, w - size) 
    y = random.randint(0, h - size) 
    outcropped = img_arr[y:y+size, x:x+size].copy() 

    # Set patch within image to 0
    img_arr[y:y+size, x:x+size] = 0

    return img_arr, outcropped



class RetinalVesselDatasetGenerative(Dataset):
  
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
    
    # Extract outcropped patch and image 
    img, outcrop = outcrop_random_patch(img)

    # Convert to torch tensors
    img = torch.from_numpy(np.expand_dims(img, 0))
    label = torch.from_numpy(np.expand_dims(outcrop, 0))

    # Create a label
    return img, label

    
if __name__ == "__main__":
    
    from torch.utils.data import DataLoader

    dataset = RetinalVesselDatasetGenerative()


    print(f"Length of dataset {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, (imgs, labels) in enumerate(dataloader):
        # Create a figure and 3 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Plot the arrays in each subplot
        img = np.array(imgs[0][0])
        label = np.array(labels[0][0])

        ax1.imshow(img)
        ax2.imshow(label)
        
        # Add titles to each subplot
        ax1.set_title('Image')
        ax2.set_title('Label')
    

        # Add a title to the entire figure
        fig.suptitle('Image with outcropped patch')


        # Show the plot
        savepath = 'reports/figures/retinalVessel/generative/' + str(i)
        plt.savefig(savepath)
    

        if i == 1:
            break