from matplotlib import pyplot as plt
from monai.inferers import sliding_window_inference

import torch

from src.models.unet_model import load_unet
from src.data.hepatic_dataset import load_hepatic_dataset

# loads dataset
data_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'
train_loader, val_loader = load_hepatic_dataset(data_path, aug=False)

# List of finetuned models
general_path = 'models/finetune/hepatic/'
model_paths = ['3drpl_finetune', 'transfer_finetune', 'random']
#models/hepatic_finetuned_e999999_k3_d0_lr1E-04_aFalse_bmm.pth <- path
# load finetuned models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_path in model_paths:

    # Get path to specific model
    path = general_path + model_path
    
    # Load model
    model = load_unet(path, device)
    model.eval() 

    # Segment on validation data and visualize
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)
            
            # plot slice [:, :, 80]
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(val_data["label"][0, 0, :, :, 80])
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")


            plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 80])
            plt.show()
            plt.savefig()
            
            # breaks after 2 iterations
            if i == 2:
                break