import cv2
import numpy as np


def loadRetinalDataset():
  """Loads and returns the retinal vessel image dataset

  Returns:
      list: list of numpy arrays corresponding to images in the dataset
  """
  dataset_dir = "DRIVE/"

  # load the training images
  train_imgs = []
  for i in range(21, 40):
    
      img_name = 'training/images/%02d_training.tif' % i
      img_path = dataset_dir + img_name
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

      # Scale images to range(0, 1)
      img = np.array(img)/255
      train_imgs.append(img)
  print("Loaded data successfully")
  
  return train_imgs

if __name__ == "__main__":
  train = loadRetinalDataset()
  
  print(f'{len(train)} images loaded')
  print(f'Dimension of data is: {np.shape(train[0])}')