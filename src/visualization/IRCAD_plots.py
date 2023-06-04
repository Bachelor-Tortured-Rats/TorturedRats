import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from src.visualization.plot_functions import plot_three_slices
import torch



    


from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    LabelFilterd,
    Rand3DElasticd,
    RandRotate90d,
    RandShiftIntensityd,
    RandZoomd
)
from src.utils.data_transformations import Addd


load_transform = Compose([LoadImaged(keys=["image", "label",'mask','label2']),
          Addd(keys=["label"],source_key='label2'),
          ScaleIntensityRanged(
            keys=["label"],
            a_min=0,
            a_max=1,
            b_min=0,
            b_max=1,
            clip=True,
        )])

ircad_path = '//zhome/cc/7/151856/Desktop/Bachelor/3Dircadb1/'

patients = [1,4,5,6,7,8,9,17]

# Defines data loaders
train_images = [f'{ircad_path}/3Dircadb1.{i}/PATIENT_DICOM/' for i in patients]
train_venoussystem = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/venoussystem/' for i in patients]
train_artery = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/artery/' for i in patients]
train_mask = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/liver/' for i in patients]
train_files = [{"image": image_name, "label": label_name, "mask": mask_name, "label2": label2_name} for image_name, label_name, mask_name, label2_name in zip(train_images, train_venoussystem, train_mask, train_artery)]




if False:
        # Save the dictionary to a file
    with open('reports/figures/Data_statistics/IRCADDataExploration.pkl', 'rb') as f:
        d1 = pickle.load(f)

    with open('reports/figures/Data_statistics/mask_IRCADDataExploration.pkl', 'rb') as f:
        d2 = pickle.load(f)

    with open('reports/figures/Data_statistics/label_IRCADDataExploration.pkl', 'rb') as f:
        d3 = pickle.load(f)

    x_img = [x for x in d1.keys()]
    y_img = [d1[key] for key in x_img]

    x_mask = [x for x in d2.keys()]
    x_mask = sorted(x_mask)
    x_mask = np.array(x_mask)
    y_mask = [d2[key] for key in x_mask]
    y_mask = np.array(y_mask)

    x_label = [x for x in d3.keys()]
    x_label = sorted(x_label)
    x_label = np.array(x_label)
    y_label = [d3[key] for key in x_label]
    y_label = np.array(y_label)

    



    cum_y = [sum(y_label[:i + 1])/sum(y_label) for i in range(len(y_label))]

    lower = x_label[sum(np.array(cum_y) < 0.001)]
    upper = x_label[-sum(np.array(cum_y) > 0.999)]
    print(len(cum_y))
    print(sum(np.array(cum_y) < 0.0005))
    print(sum(np.array(cum_y) > 0.9995))
    print(x_label==sorted(x_label))
    print(lower)
    print(upper)

    bin_width = 25

    bins = np.arange(-250, 501, bin_width)

    binned_y_mask = [sum(y_mask[np.where((b <= x_mask) & (x_mask < b + bin_width))]) for b in bins[:-1]]
    binned_y_label = [sum(y_label[np.where((b <= x_label) & (x_label < b + bin_width))]) for b in bins[:-1]]
    #[sum(y_label[x_label >= b & x_label < b + bin_width]) for b in bins[:-1]]
    
    hist_mask, edges_mask = np.histogram(binned_y_mask, bins = bins)
    hist_label, edges_label = np.histogram(binned_y_label, bins = bins)
    x = (edges_mask[1:] + edges_mask[:-1])/2
    width = bins[1]-bins[0]
    fig, ax = plt.subplots()
    ax.bar(x,binned_y_mask/sum(binned_y_mask), width = 1*width, label="Normalized liver voxel intensity count", alpha=0.5, color="blue")
    ax.bar(x,binned_y_label/sum(binned_y_label), width = 1*width, label="Normalized label voxel intensity count", alpha=0.5, color="red")
    plt.axvline(x=lower, color='yellow', linestyle='--', linewidth=2, label='Range boundaries')
    plt.axvline(x=upper, color='yellow', linestyle='--', linewidth=2)
    ax.set_xlabel('Bins')
    ax.set_ylabel('Normalized frequency')
    ax.set_title('Distributions of voxel intenities')
    ax.legend()
    plt.savefig("reports/figures/02_Data/voxel_intensities_IRCAD.png")
    plt.show()

    exit()
    #plt.bar(x_img, y_img/sum(y_img), color="blue", alpha=0.5)
    plt.bar(x_mask, y_mask/sum(y_mask), color="blue", alpha=0.5)
    plt.bar(x_label, y_label/sum(y_label), color="red", alpha=0.5)

    plt.axvline(x=lower, color='yellow', linestyle='--', linewidth=2, label='Resampling spacing in the first two dimensions')
    plt.axvline(x=upper, color='yellow', linestyle='--', linewidth=2, label='Resampling spacing in the first two dimensions')
    plt.show()
    exit()

    all_values = np.arange(-2048, 1526)

    img_voxels = [d1[i-2048.0] if  (i-2048.0) in d1 else 0 for i in range(len(all_values))]
    mask_voxels = [d2[i-2048.0] if  (i-2048.0) in d2 else 0 for i in range(len(all_values))]
    label_voxels = [d3[i-2048.0] if  (i-2048.0) in d3 else 0 for i in range(len(all_values))]



    img_hist = []
    mask_hist = []
    label_hist = []
    bin_edges = []
    # Iterate over the array in groups of 10
    jump = 20
    for i in range(0, len(all_values), jump):
        # Sum the elements in the current group
        img_sum = sum(img_voxels[i:i+jump])
        mask_sum = sum(mask_voxels[i:i+jump])
        label_sum = sum(label_voxels[i:i+jump])

        # Append the group sum to the result array
        img_hist.append(img_sum)
        mask_hist.append(mask_sum)
        label_hist.append(label_sum)

        bin_edges.append(all_values[i])

    bin_edges.append(all_values[-1])

    plt.hist(img_hist, bins=bin_edges, color="red", density=True, alpha=0.5)
    plt.hist(mask_hist, bins=bin_edges, color="blue", density=True, alpha=0.5)
    plt.hist(label_hist, bins=bin_edges, color="green", density=True, alpha=0.5)

    plt.show()

if False:
    load_dict = load_transform(train_files[4])
    print("load image")
    image = load_dict["image"]
    liver_mask = load_dict["mask"]
    label = load_dict["label"]
    print("image loaded")

    liver_mask = liver_mask > 0
    label = label > 0
    label = label * liver_mask

    #CT_image = np.rot90(CT_image, k=1, axes=(0, 1))
    #mask = np.rot90(mask, k=1, axes=(0, 1))

    dim_depths = [200, 300, 120]
    
    
    images = plot_three_slices(image,  dim_depths, label, alpha = 0.6, colors = [])

    fig = plt.figure(figsize=(6,4))
    fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)

    X = [ (2,2,1), (2,2,3), (2,2,(2,4))]
    titles = ['Sagittal plane', 'Coronal plane', 'Transverse plane']
    images[0] = np.fliplr(images[0])
    images[1] = np.fliplr(images[1])
    images[2] = np.rot90(images[2], 2)
    for i, (nrows, ncols, plot_number) in enumerate(X):
        sub = fig.add_subplot(nrows, ncols, plot_number)
        sub.imshow(images[i])
        sub.set_title(titles[i])
        sub.set_xticks([])
        sub.set_yticks([])

    plt.savefig("reports/figures/02_Data/raw_IRCAD.png")

    plt.show()

if True:
    import torch
    load_transform = Compose([
        LoadImaged(keys=["image", "label",'mask','label2']),
        EnsureChannelFirstd(keys=["image", "label",'mask','label2']),
        Addd(keys=["label"],source_key='label2'),
        CropForegroundd(keys=["image", "label"], source_key="mask"), # crops the scan to the size of the nyre
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-100,
            a_max=371,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ScaleIntensityRanged(
            keys=["label"],
            a_min=0,
            a_max=1,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest"))
    ])

    load_dict = load_transform(train_files[6])
    print("load image")
    image = torch.squeeze(load_dict["image"])
    liver_mask = torch.squeeze(load_dict["mask"])
    label = torch.squeeze(load_dict["label"])
    print("image loaded")

    """
    liver_mask = liver_mask > 0
    label = label > 0
    print(label.shape)
    print(liver_mask.shape)
    label = label * liver_mask
    """

    #CT_image = np.rot90(CT_image, k=1, axes=(0, 1))
    #mask = np.rot90(mask, k=1, axes=(0, 1))

    dim_depths = [100, 100, 100]
 
    
    images = plot_three_slices(image, dim_depths, np.zeros_like(image), alpha = 0.6, colors = [])

    fig = plt.figure(figsize=(6,4))
    fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)

    X = [ (1,3,1), (1,3,2), (1,3,3)]
    titles = ['Sagittal plane', 'Coronal plane', 'Transverse plane']
    images[0] = np.fliplr(images[0])
    images[1] = np.fliplr(images[1])
    images[2] = np.rot90(images[2], 2)
    for i, (nrows, ncols, plot_number) in enumerate(X):
        sub = fig.add_subplot(nrows, ncols, plot_number)
        print(images[i].shape)
        sub.imshow(images[i])
        sub.set_title(titles[i])
        sub.set_xticks([])
        sub.set_yticks([])

    plt.savefig("reports/figures/02_Data/preprocessed_IRCAD.png")

    plt.show()