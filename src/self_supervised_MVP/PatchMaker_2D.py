import numpy as np
import random


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

def make_sample_from_center(image, center, patch_size_inner=(30,30), patch_size_outer=(40,40)):
  
  x, y = center[0], center[1]
  patches = {}   
  count = 0 
  
  for i in [-1, 0, 1]:
    for j in [-1, 0, 1]:
      
      # Extract patch
      patch = image[x + j*(patch_size_outer[0]) : x + (j+1)*(patch_size_outer[0]),
                    y + i*(patch_size_outer[1]) : y + (i+1)*(patch_size_outer[1])]
      
      patches[count] = patch
      
      count += 1
  
  # Extract center patch
  center_patch = patches[4]
  
  # Extract random patch
  idx = np.random.randint(0,7)
  label = np.zeros(8)
  label[idx] = 1
  offset_patch = patches[idx]

  # Sample random smaller patch within larger patch
  start_row = random.randint(0, patch.shape[0] - patch_size_inner[0])
  start_col = random.randint(0, patch.shape[1] - patch_size_inner[1])

  center_patch = center_patch[start_row:start_row+patch_size_inner[0], start_col:start_col+patch_size_inner[1]]
  offset_patch = offset_patch[start_row:start_row+patch_size_inner[0], start_col:start_col+patch_size_inner[1]]

  return [center_patch, offset_patch], label
      
  
  
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