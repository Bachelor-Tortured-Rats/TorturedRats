from monai.utils import first, set_determinism

import click
import numpy as np
import matplotlib.pyplot as plt

from src.data.IRCAD_dataset import load_IRCAD_dataset
from src.data.hepatic_dataset import load_hepatic_dataset
from src.visualization.plot_functions import animate_CT


@click.command()
@click.option('-i', '--ircad_path', 'ircad_path',type=click.Path(exists=True), default='/work3/s204159/3Dircadb1/')
@click.option('-h', '--hepatic_path', 'hepatic_path',type=click.Path(exists=True),  default='/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/')
def main(ircad_path,hepatic_path):
    IRCAD_train_loader, IRCAD_val_loader = load_IRCAD_dataset(ircad_path=ircad_path,train_patients=[],val_patients=[1])
    hepatic_train_loader, hepatic_val_loader = load_hepatic_dataset(data_dir=hepatic_path,sample_size=2)

    IRCAD_val_sample = first(IRCAD_val_loader)
    hepatic_val_sample = first(hepatic_val_loader)
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15)) 
    ax[0,0].set_title("IRCAD_train_sample, layer 70")
    ax[0,0].imshow(IRCAD_val_sample["image"][0][0][:, :, 70], cmap="gray")
    ax[1,0].set_title("IRCAD_train_sample, historgram over intensity full image, excluding black pixels")
    counts, edges = np.histogram(IRCAD_val_sample["image"][0][0][:, :,:].reshape(-1), bins=30)
    ax[1,0].stairs(counts[1:], edges[1:], fill=True)
    ax[1,0].vlines(edges[1:], 0, counts[1:].max(), colors='w')
    ax[1,0].text(0.1, counts[1:].max(),f'mean: {np.mean(IRCAD_val_sample["image"][0][0][:, :,:].reshape(-1)):.4f}, std: {np.std(IRCAD_val_sample["image"][0][0][:, :,:].reshape(-1)):.4f}', size=15, color='black')
    
    ax[0,1].set_title("hepatic_train_sample, layer 66")
    ax[0,1].imshow(hepatic_val_sample["image"][0][0][:, :, 66], cmap='gray')
    ax[1,1].set_title("hepatic_train_sample, historgram over intensity full image, excluding black pixels")
    counts, edges = np.histogram(hepatic_val_sample["image"][0][0][:, :, :].reshape(-1), bins=30)
    ax[1,1].stairs(counts[1:], edges[1:], fill=True)
    ax[1,1].vlines(edges[1:], 0, counts[1:].max(), colors='w')
    ax[1,1].text(0.1, counts[1:].max(),f'mean: {np.mean(hepatic_val_sample["image"][0][0][:, :,:].reshape(-1)):.4f}, std: {np.std(hepatic_val_sample["image"][0][0][:, :,:].reshape(-1)):.4f}', size=15, color='black')

    #animate_CT(IRCAD_val_sample["image"][0][0][:, :, :] , angle = 2, filename="reports/figures/visualization/IRCAD_val_sample")
    #animate_CT(hepatic_val_sample["image"][0][0][:, :, :] , angle = 2, filename="reports/figures/visualization/hepatic_val_sample")
    
    fig.savefig('reports/figures/visualization/IRCADandHepaticDistribution.png')

if __name__ == "__main__":
    main()