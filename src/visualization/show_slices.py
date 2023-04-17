import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from PyPDF2 import PdfMerger
import io

import torch

from src.models.unet_model import load_unet
from src.data.hepatic_dataset import load_hepatic_dataset

# loads dataset
data_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'
train_loader, val_loader = load_hepatic_dataset(data_path, train_label_proportion=.1)

# List of finetuned models
general_path = 'models/finetune/hepatic/'

# UPDATE WHEN WE HAVE THE SPECIFIC MODEL PATHS
model_paths = ['3drpl_finetune', 'transfer_finetune', 'random']
#models/hepatic_finetuned_e999999_k3_d0_lr1E-04_aFalse_bmm.pth <- path

# load finetuned models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# For now the random model is just hardcoded
path = "models/hepatic_finetuned_e999999_k3_d0.1_lr1E-04_aFalse_bmm.pth"
#path = general_path + 

# Load model
model, params = load_unet(path, device)
model.eval() 

# Initialize a PDF merger
pdf_merger = PdfMerger()

# Segment on validation data and visualize
with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        
        # Get output
        val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)
        
        # Generate a plot of each slice
        for depth, slice in enumerate(val_outputs[0, 0, :, :, :]):
            
            # Create a figure
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
            
            # Plot the image
            ax1.imshow(val_data["image"][0, 0, :, :, depth], cmap="gray")
            ax1.set_title("Original Image")
            
            # Plot the ground truth mask
            ax2.imshow(val_data["label"][0, 0, :, :, depth])
            ax2.set_title("Ground Truth Mask")
            
            # Plot the model prediciction
            ax3.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, depth])
            ax3.set_title("Model Prediction")
            
            # Set the main title
            fig.suptitle(f'Image {i} - Depth {depth}')

            
            pdf_bytes = io.BytesIO()
            fig.savefig(pdf_bytes, format='pdf')
            pdf_merger.append(pdf_bytes)
            plt.close()
            
            print(f"Slice {depth} of {len(val_outputs[0, 0, :, :, :])}")

        # breaks after 5 iteration
        if i == 1:
            break

# Save the pdf
save_path = 'reports/figures/inference/transfer_5_first.pdf'
with open(save_path, 'wb') as output:
    pdf_merger.write(output)

print(f"Succesfully saved results in \n{save_path}")
 #plt.savefig(f'/reports/figures/finetune_wrt_labelproportion{model_path}')
