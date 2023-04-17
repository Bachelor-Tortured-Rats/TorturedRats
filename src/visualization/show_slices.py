import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from PyPDF2 import PdfMerger
import io
import pdb
import torch
from src.models.unet_model import load_unet
from src.data.hepatic_dataset import load_hepatic_dataset


def extract_slicewise_output_to_pdf(model_path = None, model_type = 'transfer', num_images = 5):
    
    '''
    Makes and saves segmentation predictions into a pdf file
    with each page corresponding to a slice.
    
    Input Args:
    model       - string:   A string indicating what model to load
    num_images  - int:      The number of validation images to use
    '''
    
    if model_path is None:
        print("No path provided")
        return
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, params = load_unet(model_path, device)
    model.eval()
    
    # Initialize a PDF merger
    pdf_merger = PdfMerger()

    # Infer on validation data and visualize results
    with torch.no_grad():
        
        # Loop over elements in the validation dataset
        for i, val_data in enumerate(val_loader):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            
            # Get output
            val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)
            
            # Generate a plot of each slice 
            for depth in range(val_outputs.shape[-1]):
                
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

                # Save the current page to the pdf merger
                pdf_bytes = io.BytesIO()
                fig.savefig(pdf_bytes, format='pdf')
                pdf_merger.append(pdf_bytes)
                plt.close()
                
                # Print status
                print(f"Model Type: {model_type} | Image {i+1} of {num_images} | Slice {depth+1} of {val_outputs.shape[-1]} saved")

            # breaks after specified number of iteration
            if i+1 == num_images:
                break

    # Merge and save the pdf
    save_path = f'reports/figures/inference/{model_type}_{num_images}_first.pdf'
    with open(save_path, 'wb') as output:
        pdf_merger.write(output)

    print(f"Succesfully saved results in \n{save_path}")


if __name__ == "__main__":
    
    # loads dataset
    data_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'
    train_loader, val_loader = load_hepatic_dataset(data_path, train_label_proportion=.1)

    # models
    models = ['3drpl', 
              'transfer', 
              'random']


    for model in models:
        
        path = f'models/finetuned/hepatic_{model}_steps_1000.pth'
        
        extract_slicewise_output_to_pdf(model_path = path, model_type=model, num_images=5)