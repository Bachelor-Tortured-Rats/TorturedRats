import numpy as np
import random
from monai.transforms import CropForeground, RandSpatialCrop
import torch

def generate_patch_pair_MONAI(img, outer_patch_width, inner_patch_width):

  # Add channel dimension
  #img = np.expand_dims(img, axis=0)
  img = np.transpose(img, (2, 0, 1))
  
  # Crop foreground with temporary crop function
  crop_fn = lambda x: x > 0.5
  transform1 = CropForeground(crop_fn)
  img = transform1(img)

  # Randomly select one of 8 patch locations
  patch_location = random.randint(0, 7)

  # Multipliers for the height and width of the first zoom-in crop
  primary_crop_multiplier_dict = {0: [2, 2], 1: [2, 1], 2: [2, 2], 3: [1, 2], 4: [1, 2], 5: [2, 2], 6: [2, 1], 7: [2, 2]}
  primary_crop_multiplier = primary_crop_multiplier_dict[patch_location]
  primary_crop_shape = [outer_patch_width * primary_crop_multiplier[0], outer_patch_width * primary_crop_multiplier[1]]
  
  # Crop large patch larger than outer patch
  primary_crop = RandSpatialCrop(primary_crop_shape, random_size=False)
  primary_crop.set_random_state(42)
  img = primary_crop(img)

  center_patch = img
  offset_patch = img

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
  secondary_crop = RandSpatialCrop([inner_patch_width, inner_patch_width], random_size=False)

  #secondary_crop.set_random_state(42)
  center_patch = secondary_crop(center_patch)
  offset_patch = secondary_crop(offset_patch)

  """
  # Create label
  label = np.zeros(8)
  label[patch_location] = 1
  """

  return center_patch, offset_patch, patch_location


''' THIS IS THE FUNCTION WE WILL PROBABLY USE'''
def extract_patches_and_label_from_center(image, center, patch_size_inner=(30,30), patch_size_outer=(40,40)):
  """ Extracts a center patch, a random patch surounding the center patch, 
      as well as the one hot encoded label from an image. 

  Args:
      image (numpy ndarray): a 2D image
      center (tuple): coordinates of the pixel in the center of the center patch
      patch_size_inner (tuple, optional): size of the actual patches. Defaults to (30,30).
      patch_size_outer (tuple, optional): size of the outer patch from which a random subpatch is sampled. Defaults to (40,40).

  Returns:
      list[numpy ndarray, numpy ndarray], list[int]: first list contains the patches, second list is the one hot encoded label
  """
  
  y, x = center[0], center[1]
  patch_locations = {}
  patches = {}   
  count = 0 
  half_patch_size = patch_size_outer[0] // 2
  for i in [-1, 0, 1]:
    for j in [-1, 0, 1]:
      
      # Extract patch
      patch = image[y + i*(patch_size_outer[0]) - half_patch_size : y + i*(patch_size_outer[0]) + half_patch_size,
                    x + j*(patch_size_outer[1]) - half_patch_size : x + j*(patch_size_outer[1]) + half_patch_size]
      patches[count] = patch
      patch_locations[count] = [(y + i*(patch_size_outer[0]) - half_patch_size, x + j*(patch_size_outer[1]) - half_patch_size), 
                                (y + i*(patch_size_outer[0]) + half_patch_size, x + j*(patch_size_outer[1]) + half_patch_size)]
      print(patch_locations)
      count += 1
  
  # Extract center patch and remove it from 
  center_patch = patches[4]
  
  # Extract random patch from remaining ones
  # idx = np.random.randint(0,7)
  idx = random.choice([0, 1, 2, 3, 5, 6, 7])
  offset_patch = patches[idx]

  # Extract label on chosen patch
  label = np.zeros(9) # 9 for now, should be 8
  label[idx] = 1
  offset_patch = patches[idx]

  # Sample random smaller patch within larger patch
  start_row = random.randint(0, patch.shape[0] - patch_size_inner[0])
  start_col = random.randint(0, patch.shape[1] - patch_size_inner[1])

  center_patch = center_patch[start_row:start_row+patch_size_inner[0], start_col:start_col+patch_size_inner[1]]
  offset_patch = offset_patch[start_row:start_row+patch_size_inner[0], start_col:start_col+patch_size_inner[1]]

  return [center_patch, offset_patch], label, patch_locations, patches
  
  
''' THE FOLLOWING ARE UTILITY FUNCTIONS WE MIGHT USE LATER BUT NOT ATM'''
def extract_2D_patches(image, patch_size=(4,4)):
  """
  Extracts all 2D patches of the given patch size from a 2D image.
  
  Args:
    image: 2d image 
    patch_size: tuple with 2 elements
  
  Returns:
    dictionary: keys are patch coordinates and values are patch parts of image
  """

  patches = {}
  x_patches, y_patches = image.shape[0] // patch_size[0], image.shape[1] // patch_size[1]
  
  for y in range(y_patches):
    for x in range(x_patches):
      patch = image[x*patch_size[0]:(x+1)*patch_size[0], y*patch_size[1]:(y+1)*patch_size[1]]
      patches[(x, y)] = patch

  return patches 

def extract_patch_labels_from_centerpatch(patches, centerPatchCoodinates):
  """Creates and returns a dictionary of labels for patches around a center patch 

  Args:
      patches (dictionary): a dictionary of patches, keys are coordinates and values are corresponding part of image
      centerPatchCoodinates (tuple): tuple with coordinate of the center patch, in patch coordinates

  Returns:
      dictionary: keys are labels and values are patches
  """
  
  patch_labels = {}
  x, y = centerPatchCoodinates[0], centerPatchCoodinates[1]
  
  
  patch_labels['center'] = patches[(x,y)]
  
  count = 1
  for i in [-1, 0, 1]:
    for j in [-1, 0, 1]:
      if i == 0 and j == 0:
        continue
      
      px = x + i
      py = y + j
      
      patch_labels[count] = patches[(px, py)]
      
      count += 1
      
  return patch_labels

def sample_random_smaller_patch(patch, smaller_patch_size = (3,3)):

  sample_height, sample_width = smaller_patch_size[0], smaller_patch_size[1]    # size of the smaller array to sample

  start_row = random.randint(0, patch.shape[0] - sample_height)
  start_col = random.randint(0, patch.shape[1] - sample_width)

  return patch[start_row:start_row+sample_height, start_col:start_col+sample_width]

def make_sample(patches_with_labels, smaller_patch_size = (3,3)):

  center_patch = patches_with_labels['center']
  idx = np.random.randint(0,7)
  label = np.zeros(8)
  label[idx] = 1
  random_patch = patches_with_labels[idx+1]

  return sample_random_smaller_patch(center_patch, smaller_patch_size=smaller_patch_size), sample_random_smaller_patch(random_patch, smaller_patch_size=smaller_patch_size), label



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

  from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForeground,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    LabelFilterd,
    Rand3DElasticd,
    RandRotate90d,
    RandShiftIntensityd,
    RandZoomd,
    RandSpatialCrop
  )

  #img = np.expand_dims(img, axis=0)

  patch_center, patch_offset, label = generate_patch_pair_MONAI(img, outer_patch_width=80, inner_patch_width=60)
  
  print(label)

  

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