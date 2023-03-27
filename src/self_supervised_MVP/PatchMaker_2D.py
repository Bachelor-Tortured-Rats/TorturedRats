import numpy as np
import random
from monai.transforms import CropForeground, RandSpatialCrop
import torch

def generate_patch_pair_MONAI(img, outer_patch_width, inner_patch_width, num_pairs=1):

  # Add channel dimension
  #img = np.expand_dims(img, axis=0)
  img = np.transpose(img, (2, 0, 1))
  
  # Crop foreground with temporary crop function
  crop_fn = lambda x: x > 0.5
  foreground_crop = CropForeground(crop_fn)

  # Crop foreground
  img = foreground_crop(img)

  # Randomly select one of 8 patch locations n times
  patch_locations = np.random.randint(0, 7, num_pairs)
  #patch_location = random.randint(0, 7)
  
  # Multipliers for the height and width of the first zoom-in crop of each patch location
  primary_crop_multiplier_dict = {0: [2, 2], 1: [2, 1], 2: [2, 2], 3: [1, 2], 4: [1, 2], 5: [2, 2], 6: [2, 1], 7: [2, 2]}
  primary_crop_multipliers = [primary_crop_multiplier_dict[patch_location] for patch_location in patch_locations]
  #primary_crop_multiplier = primary_crop_multiplier_dict[patch_location]
  primary_crop_shapes = [[outer_patch_width * primary_crop_multiplier[0], outer_patch_width * primary_crop_multiplier[1]] for primary_crop_multiplier in primary_crop_multipliers]
  #primary_crop_shape = [outer_patch_width * primary_crop_multiplier[0], outer_patch_width * primary_crop_multiplier[1]]
  
  # Initialize the center and offset patches for stacking image patches
  center_patches = []
  offset_patches = []

  for i, primary_crop_shape in enumerate(primary_crop_shapes):
    # Define the primary and secondary crop functions
    primary_crop = RandSpatialCrop(primary_crop_shape, random_size=False)
    secondary_crop = RandSpatialCrop([inner_patch_width, inner_patch_width], random_size=False)

    # Crop large patch larger than outer patch
    #primary_crop.set_random_state(42)
    img = primary_crop(img)

    center_patch = img
    offset_patch = img

    # Get the patch location
    patch_location = patch_locations[i]

    # Cut out the center and offset patches
    if patch_location <= 4:
      offset_patch = offset_patch[:, :outer_patch_width, :]
    else:
      offset_patch = offset_patch[:, -outer_patch_width:, :]
    
    if patch_location in [0, 1, 3, 5, 6]:
      offset_patch = offset_patch[:, :, :outer_patch_width]
    else:
      offset_patch = offset_patch[:, :, -outer_patch_width:]

    if patch_location >= 3:
      center_patch = center_patch[:, :outer_patch_width, :]
    else:
      center_patch = center_patch[:, -outer_patch_width:, :]
    
    if patch_location not in [0, 3, 5]:
      center_patch = center_patch[:, :, :outer_patch_width]
    else:
      center_patch = center_patch[:, :, -outer_patch_width:]

    # Secondary crop to get inner patches
    #secondary_crop.set_random_state(42)
    center_patch = secondary_crop(center_patch)
    offset_patch = secondary_crop(offset_patch)

    # Stack the patches
    center_patches.append(center_patch)
    offset_patches.append(offset_patch)

  """
  # Create label
  label = np.zeros(8)
  label[patch_location] = 1
  """
  #print(center_patch.shape)
  #print(type(patch_location))
  return center_patches, offset_patches, patch_locations


if __name__ == "__main__":
  
  import os
  import pandas as pd
  from torchvision.io import read_image
  import cv2
  import numpy as np
  from PatchMaker_2D import *
  from torch.utils.data import Dataset
  import torch
  import matplotlib.pyplot as plt
  
  img_path = ['DRIVE/training/images/%02d_training.tif' % i for i in range(21,40)][1]

  img = cv2.imread(img_path)
  img = np.array(img)/255


  #img = np.expand_dims(img, axis=0)

  patch_centers, patch_offsets, labels = generate_patch_pair_MONAI(img, outer_patch_width=80, inner_patch_width=60, num_pairs=10)
  
  print(labels)
  print(torch.stack(patch_centers).shape)
  print(torch.stack(patch_offsets).shape)

  exit()
  

  #print(img == torch.squeeze(patch_center).numpy())

  def find_patch_box(arr, patch):
    """
    Given a 2D array `arr` and a patch of that array `patch`,
    return the coordinates of the box that bounds the patch.
    """
    m, n = arr.shape
    pm, pn = patch.shape
    for i in range(m - pm + 1):
        for j in range(n - pn + 1):
            if (arr[i:i+pm, j:j+pn] == patch).all():
                return (i, j, i+pm-1, j+pn-1)
    return None

  center_cords = find_patch_box(img[:,:,0], torch.squeeze(patch_center).numpy()[0,:,:])
  offset_cords = find_patch_box(img[:,:,0], torch.squeeze(patch_offset).numpy()[0,:,:])

  img = img*255
  center_patch = torch.squeeze(patch_center).numpy() * 255
  offset_patch = torch.squeeze(patch_offset).numpy() * 255

  print(center_patch.shape)
  center_patch = np.transpose(center_patch, (1,2,0))
  offset_patch = np.transpose(offset_patch, (1,2,0))

  print(center_cords)
  print(offset_cords)

  cv2.rectangle(img, (center_cords[1], center_cords[0]), (center_cords[3], center_cords[2]), (0, 0, 255), 2)
  cv2.rectangle(img, (offset_cords[1], offset_cords[0]), (offset_cords[3], offset_cords[2]), (200, 200, 0), 2)

  # Create a figure and 3 subplots
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

  # Plot the arrays in each subplot
  ax1.imshow(np.flip(img.astype(np.uint8), axis=2))

  #patch_center[0,0] = np.max(img)
  #patch_offset[0,0] = np.max(img)
  ax2.imshow(np.flip(center_patch.astype(np.uint8), axis=2))
  ax3.imshow(np.flip(offset_patch.astype(np.uint8), axis=2))
  # Show the plot
  savepath = 'reports/figures/retinalVessel/' + "LilleTester2"
  plt.savefig(savepath)

  # A function that returns the box coordinates of a patch in an image
  def find_patch_coordinates(patch, image):
    x, y = np.where(np.all(image == patch, axis=(0, 1))) #np.where(image == patch)
    return (x[0], y[0]), (x[-1], y[-1])
  #b1, b2 = find_patch_coordinates(patch_center.numpy()[0,:,:], img)
  #print(b1)
  #print(b2)


  """
  print("------ Tests the function -------")

  arr = np.arange(1, 15*15+1).reshape((15, 15))
  print(f'Test array: \n{arr}')

  data, label = make_sample_from_center(image=arr, center=(6,6), patch_size_inner=(2,2), patch_size_outer=(4,4))
  print(f'label is: {label}')
  print(f'data is: {data}')

  """

  '''
  # Create all patches
  patches = extract_2D_patches(arr)

  # Provide labels for 
  patch_labels = extract_patch_labels_from_centerpatch(patches, centerPatchCoodinates=(1,1))

  # Make a sample
  center, patch, label = make_sample(patch_labels)
  
  
  print(f'Center patch is: {center}')
  print(f'patch is: {patch}')
  print(f'label is: {label}')
  '''