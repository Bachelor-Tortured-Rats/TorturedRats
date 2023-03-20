import numpy as np
import random

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
  label = np.zeros(9)
  label[idx] = 1
  offset_patch = patches[idx]

  # Sample random smaller patch within larger patch
  start_row = random.randint(0, patch.shape[0] - patch_size_inner[0])
  start_col = random.randint(0, patch.shape[1] - patch_size_inner[1])

  center_patch = center_patch[start_row:start_row+patch_size_inner[0], start_col:start_col+patch_size_inner[1]]
  offset_patch = offset_patch[start_row:start_row+patch_size_inner[0], start_col:start_col+patch_size_inner[1]]

  return [center_patch, offset_patch], label, patch_locations
  
  
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

  
  print("------ Tests the function -------")

  arr = np.arange(1, 15*15+1).reshape((15, 15))
  print(f'Test array: \n{arr}')

  data, label = make_sample_from_center(image=arr, center=(6,6), patch_size_inner=(2,2), patch_size_outer=(4,4))
  print(f'label is: {label}')
  print(f'data is: {data}')

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