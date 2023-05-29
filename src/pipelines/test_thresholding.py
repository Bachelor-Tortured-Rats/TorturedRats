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

    mask = torch.logical_and(image >= lower_threshold, image <= upper_threshold)[0,:,:,:]
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


# Store results
dice_scores = {}   
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Loop over each label proportion
for label_proportion in [.01, .02, .03, .05, .07, .1, 1.0]:

  for fold in range(5):
    print(f"Finding optimal threshold for lp {label_proportion} and fold {fold}")
    dice_results = []
    #pdb.set_trace()
    train_loader, val_loader, test_loader = load_hepatic_dataset(
                data_dir = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/',
                k_fold=fold, 
                setup='threshold', 
                train_label_proportion=label_proportion)

    # Train (find best thresholding split)
    # Set the range and step size for the threshold values
    lower_threshold_range = np.arange(0.1, 0.9, 0.1)
    upper_threshold_range = np.arange(0.1, 0.9, 0.1)

    best_lower_threshold = None
    best_upper_threshold = None
    best_dice_coefficient = -1.0  # Initialize with a low value

    for lower_threshold in lower_threshold_range:
        for upper_threshold in upper_threshold_range:
            dice_coefficient_sum = 0.0
            num_images = 0

            for batch_data in train_loader:
                inputs, labels = (
                    batch_data["image"].view(-1, 1, *batch_data["image"].shape[-3:]),
                    batch_data["label"].view(-1, 1, *batch_data["label"].shape[-3:]),
                )

                # Iterate over each image in the batch
                for image, label in zip(inputs, labels):
                    # Convert the image and label to numpy arrays
                    image = image.numpy()
                    label = label.numpy()

                    # Assuming the label and background values are represented as 1 and 0, respectively
                    # Apply the thresholding to obtain the binary prediction
                    binary_prediction = threshold_3d_image(image, lower_threshold=lower_threshold, upper_threshold=upper_threshold)
                    #binary_prediction = np.logical_and(image >= lower_threshold, image <= upper_threshold)

                    # Compute the Dice coefficient between the binary prediction and the ground truth label
                    dice_coefficient = dice_score(binary_prediction, label[0])  

                    dice_coefficient_sum += dice_coefficient
                    num_images += 1

            # Calculate the average Dice coefficient for the current threshold pair
            average_dice_coefficient = dice_coefficient_sum / num_images

            # Update the best threshold pair if the average Dice coefficient is improved
            if average_dice_coefficient > best_dice_coefficient:
                best_dice_coefficient = average_dice_coefficient
                best_lower_threshold = lower_threshold
                best_upper_threshold = upper_threshold
   
    print(f"Found values: {best_lower_threshold} and {best_upper_threshold}")
    print("Starts Testing")
    # we evaluate the model with the found threshold values
    for test_data in test_loader:
      
      prediction = threshold_3d_image(test_data['image'], lower_threshold=best_lower_threshold, upper_threshold=best_upper_threshold)
      actual_label = test_data["label"][0, 0, :, :,:]
      
      #pdb.set_trace()
      
      # compute metric for current iteration
      dice_results.append(dice_score(prediction, actual_label))
      #dice_metric(y_pred=prediction, y=actual_label)
    #pdb.set_trace()
    # aggregate the final mean dice result
    #dice_metric_value = dice_metric.aggregate().item()
    
    # Compute average dice score on test set
    avg_dice = np.mean(dice_results)
    std_err_dice = np.std(dice_results, ddof=1) / np.sqrt(np.size(dice_results))
    dice_scores[(label_proportion, fold)] = (avg_dice, std_err_dice)
    print(f"Average test dice score of lp {label_proportion} and fold {fold}: {avg_dice}")

save_dictionary(dice_scores, 'thresholdingresults_2.pkl')