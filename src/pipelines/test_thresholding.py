from src.data.hepatic_dataset import load_hepatic_dataset
from monai.metrics import DiceMetric
import numpy as np
import pdb 
import pickle
import numpy as np
import torch

def threshold_3d_image(image, lower_threshold, upper_threshold):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if image.dim() == 3:  # Single channel 3D image
        image = image.unsqueeze(0)  # Add batch dimension

    mask = torch.logical_and(image >= lower_threshold, image <= upper_threshold)[0,0,:,:,:]
    return mask.byte()
  
def dice_score(y_pred, y):
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    if y_pred.dim() == 4:  # Remove batch dimension if present
        y_pred = y_pred.squeeze(0)

    if y.dim() == 4:  # Remove batch dimension if present
        y = y.squeeze(0)

    intersection = torch.sum(y_pred * y)
    union = torch.sum(y_pred) + torch.sum(y)

    dice = (2.0 * intersection) / (union + 1e-8)
    return dice.item()
  
def save_dictionary(dictionary, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)



# Set threshold values
UPPER_THRESHOLD = .7
LOWER_THRESHOLD = .6

# Store results
dice_scores = {}   
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Loop over each label proportion
for label_proportion in [.01, .02, .03, .05, .07, .1, 1.0]:

  for fold in range(5):
    
    dice_results = []
    train_loader, val_loader = load_hepatic_dataset(
                data_dir = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/',
                k_fold=fold, 
                setup='thresholding', 
                train_label_proportion=label_proportion)

    # we evaluate the model every val_interval epochs
    for val_data in val_loader:
      
      prediction = threshold_3d_image(val_data['image'], lower_threshold=LOWER_THRESHOLD, upper_threshold=UPPER_THRESHOLD)
      actual_label = val_data["label"][0, 0, :, :,:]
      
      #pdb.set_trace()
      
      # compute metric for current iteration
      dice_results.append(dice_score(prediction, actual_label))
      #dice_metric(y_pred=prediction, y=actual_label)
    #pdb.set_trace()
    # aggregate the final mean dice result
    #dice_metric_value = dice_metric.aggregate().item()
    
    dice = sum(dice_results)
    dice_scores[(label_proportion, fold)] = dice
    
    # reset the status for next validation round
    #dice_metric.reset()
print(dice_scores)
save_dictionary(dice_scores, 'thresholdingresults.pkl')