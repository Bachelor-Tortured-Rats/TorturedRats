import nibabel as nib
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

# specify the path to the NIfTI file
ircad_path = '//zhome/cc/7/151856/Desktop/Bachelor/3Dircadb1/'

patients = [1,4,5,6,7,8,9,17]

# Defines data loaders
train_images = [f'{ircad_path}/3Dircadb1.{i}/PATIENT_DICOM/' for i in patients]
train_venoussystem = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/venoussystem/' for i in patients]
train_artery = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/artery/' for i in patients]
train_mask = [f'{ircad_path}/3Dircadb1.{i}/MASKS_DICOM/liver/' for i in patients]
train_files = [{"image": image_name, "label": label_name, "mask": mask_name, "label2": label2_name} for image_name, label_name, mask_name, label2_name in zip(train_images, train_venoussystem, train_mask, train_artery)]



import pandas as pd
# for loop to collect data and add to dataframe
df = pd.DataFrame(columns=["filename", "dim0", "dim1", "dim2", "pixdim0", "pixdim1", "pixdim2"])
df_intensities = {}
df_liver_intensities = {}
df_labels_intensities = {}



for i in range(len(train_images)):
    print(i+1, "/", len(train_images))
    #print({"image": train_images[i], "label": train_masks[i]})
    print("Make load dict")
    load_dict = load_transform(train_files[i])

    
    print("load image")
    image = load_dict["image"]
    liver_mask = load_dict["mask"]
    label = load_dict["label"]
    print("image loaded")

    liver_mask = liver_mask > 0
    label = label > 0

    """
    #[liver_mask.flatten()!=min(liver_mask.flatten())]
    unique_nums_image, counts_image = np.unique(image.flatten(), return_counts=True)
    print("Disp1")
    plt.bar(unique_nums_image, counts_image/sum(counts_image), alpha=0.5)
    #plt.hist(image.flatten(), density=True)

    liver_filtered = image*liver_mask
    liver_filtered = liver_filtered[liver_filtered!=0].flatten()
    unique_nums_liver, counts_liver = np.unique(liver_filtered, return_counts=True)
    print("Disp2")
    plt.bar(unique_nums_liver, counts_liver/sum(counts_liver), alpha=0.5)
    #plt.hist(liver_filtered[liver_filtered!=0].flatten(), color = "red", density=True)


    label_filtered = image*label
    label_filtered = label_filtered[label_filtered!=0].flatten()
    unique_nums_labal, counts_label = np.unique(label_filtered, return_counts=True)
    print("Disp3")
    plt.bar(unique_nums_labal, counts_label/sum(counts_label), alpha=0.5)
    #plt.hist(label_filtered[label_filtered!=0].flatten(), color = "green", density=True)
    
    plt.show()
    exit()
    """
    

    """
    fake_mask = np.ones_like(image)*0
    fake_mask[200:300, 200:300, 40:60] = 1

    
    plts = plot_three_slices(image,  [250,259, 60], liver_mask, alpha = 0.6, colors = (0,0,1))
    plt.imshow(torch.sum(liver_mask, dim=2))
    plt.show()
    print(np.unique(liver_mask))
    plt.imshow(plts[0])
    plt.show()
    plt.imshow(plts[1])
    plt.show()
    plt.imshow(plts[2])
    plt.show()
    exit()
    """

    pixdims = load_dict['image_meta_dict']['spacing']
    shapes = image.shape

    df = df.append({"filename": f"patient_{patients[i]}", "dim0": shapes[0], "dim1": shapes[1], "dim2": shapes[2], "pixdim0": pixdims[0], "pixdim1": pixdims[1], "pixdim2": pixdims[2]}, ignore_index=True)

    unique_nums, counts = np.unique(image.flatten(), return_counts=True)
    occurrences = dict(zip(unique_nums, counts))

    liver_filtered = image*liver_mask
    liver_filtered = liver_filtered[liver_filtered!=0].flatten()
    unique_nums_liver, counts_liver = np.unique(liver_filtered, return_counts=True)
    occurrences_liver = dict(zip(unique_nums_liver, counts_liver))

    label_filtered = image*label
    label_filtered = label_filtered[label_filtered!=0].flatten()
    unique_nums_label, counts_label = np.unique(label_filtered, return_counts=True)
    occurrences_label = dict(zip(unique_nums_label, counts_label))

    new_df_intensities = {**df_intensities, **occurrences}
    new_df_liver_intensities = {**df_liver_intensities, **occurrences_liver}
    new_df_labels_intensities = {**df_labels_intensities, **occurrences_label}

    for key, value in df_intensities.items():
        if key in occurrences:
            new_df_intensities[key] = df_intensities[key] + occurrences[key]

        if (key in occurrences_liver) and (key in df_liver_intensities):
            new_df_liver_intensities[key] = df_liver_intensities[key] + occurrences_liver[key]

        if (key in occurrences_label) and (key in df_labels_intensities):
            new_df_labels_intensities[key] = df_labels_intensities[key] + occurrences_label[key]
    
    df_intensities = new_df_intensities
    df_liver_intensities = new_df_liver_intensities
    df_labels_intensities = new_df_labels_intensities


#df.to_csv('reports/figures/Data_statistics/hepaticDataExploration.csv', index=False)

df.to_csv('reports/figures/Data_statistics/IRCADDataExploration.csv', index=False)

import pickle


# Save the dictionary to a file
with open('reports/figures/Data_statistics/IRCADDataExploration.pkl', 'wb') as f:
    pickle.dump(df_intensities, f)

with open('reports/figures/Data_statistics/mask_IRCADDataExploration.pkl', 'wb') as f:
    pickle.dump(df_liver_intensities, f)

with open('reports/figures/Data_statistics/label_IRCADDataExploration.pkl', 'wb') as f:
    pickle.dump(df_labels_intensities, f)