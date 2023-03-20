import os
import pandas as pd
from torchvision.io import read_image
import cv2
import numpy as np
from PatchMaker_2D import *

class RetinalVesselDataset(Dataset):
  
  def __init__(self):
    
    # load the paths to the training images 
    self.paths = ['DRIVE/training/images/%02d_training.tif' % i for i in range(21,40)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
      
      # Load image from path
      img_path = self.paths[idx]
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
      img = np.array(img)/255

      
      # Create all patches
      patches = extract_2D_patches(img, patch_size=(40,40))

      # Provide labels for 
      patch_labels = extract_patch_labels_from_centerpatch(patches, centerPatchCoodinates=(3,3))

      # Make a sample
      center, patch, label = make_sample(patch_labels, smaller_patch_size=(30,30))
      data = {'center': center, 'offset': patch}

      # Create a label
      return data, label