import numpy as np


def extract_2D_patches(image, patch_size=(2,2)):
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


if __name__ == "__main__":
  
  image = np.random.rand(10,10) * 255

  arr = np.arange(1, 101).reshape((10, 10))
  print(f'Test array: \n{arr}')
  
  patches = extract_2D_patches(arr)
  patch_labels = extract_patch_labels_from_centerpatch(patches, centerPatchCoodinates=(2,2))

  for key, value in patch_labels.items():
    print(f'Patch label: {key}\n{value}')