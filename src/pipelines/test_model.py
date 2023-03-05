from monai.utils import set_determinism
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


import torch
import matplotlib.pyplot as plt
import argparse

from src.utils.data_transformations import Addd
from src.visualization.plot_functions import animate_CT
from src.models.unet_model import load_unet
from src.data.IRCAD_dataset import load_IRCAD_dataset


def test_model(model, data_loader, device, data_saving_path="reports/figures/validator", show_inference=False, save_animation=False):
    """Evaluates a unet pytorch model

    Args:
        model (pytorch): pytorch unet model
        data_loader (data_loader): data loader
        data_saving_path (str, optional): folder to save images. Defaults to "figures/validator".
        show_inference (bool, optional): save image of inference. Defaults to False.
        save_animation (bool, optional): save animation of inference. Defaults to False.
    """

    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    with torch.no_grad():
        for i, val_data in enumerate(data_loader):
            
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            
            val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)

            dice_metric(y_pred=torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :,:], y=val_data["label"][0, 0, :, :,:])
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            print('dice score ', metric)
            # reset the status for next validation round
            dice_metric.reset()

            if save_animation:
                animate_CT(240*(val_data["image"][0, 0, :, :, :]+6)/12 , angle = 2, masks = [val_data["label"][0, 0, :, :,:], torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :,:]], alpha=0.2, filename="figures/validator/slices")
            

            if show_inference:
                plt.figure(f"check dice score: {metric}", (18, 12))
                
                plt.subplot(1, 3, 1)
                plt.title(f"image {i} dice score: {metric}, {val_data['image_meta_dict']['filename_or_obj']}")
                plt.imshow(val_data["image"][0, 0, :, :, 60], cmap="gray")
                plt.subplot(1, 3, 2)
                plt.title(f"label {i}")
                plt.imshow(val_data["label"][0, 0, :, :, 60])
                plt.subplot(1, 3, 3)
                plt.title(f"output {i}")
                plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 60])
                
                plt.savefig(f'{data_saving_path}/validation_{i}.png')

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='data_path', type=str, help="A string argument")
    parser.add_argument(dest='model_path', type=str, help="An integer argument")
    parser.add_argument('--animation', action="store_true", default=False)
    parser.add_argument('--inference', action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    set_determinism(seed=420)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = arg_parser()

    val_loader = load_IRCAD_dataset(ircad_path=args.data_path,patients_val=[1,4])
    unet =  load_unet(model_path= args.model_path, device=device)
    test_model(unet, val_loader, device,save_animation=args.animation,show_inference=args.inference)
