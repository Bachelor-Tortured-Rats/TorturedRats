import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt

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

load_transform = Compose([LoadImaged(keys=["image", "label"])])

# specify the path to the NIfTI file
data_dir = '/dtu/3d-imaging-center/projects/2020_QIM_22_rat_kidney/analysis/study_diabetic'

rats = [21, 22, 24,25,28,33,36,37,38,43,47,48,51,52,54,57]

train_images = [f'{data_dir}/aligned/rat{i}_aligned_rigid.nii' for i in rats]
train_masks = [f'{data_dir}/maskKidney/rat{i}_kidneyMaskProc.nii.gz' for i in rats]



import pandas as pd
# for loop to collect data and add to dataframe
df = pd.DataFrame(columns=["filename", "dim0", "dim1", "dim2", "pixdim0", "pixdim1", "pixdim2"])
df_intensities = {}
df_intensities_kidney = {}



for i in range(len(train_images)):
    print(i+1, "/", len(train_images))
    #print({"image": train_images[i], "label": train_masks[i]})
    load_dict = load_transform({"image": train_images[i], "label": train_masks[i]})

    
    image = load_dict["image"]
    mask = load_dict["label"]
    pixdims = load_dict['image_meta_dict']['pixdim'][1:4]
    shapes = image.shape

    df = df.append({"filename": f"rat{rats[i]}_aligned_rigid.nii", "dim0": shapes[0], "dim1": shapes[1], "dim2": shapes[2], "pixdim0": pixdims[0], "pixdim1": pixdims[1], "pixdim2": pixdims[2]}, ignore_index=True)

    unique_nums, counts = np.unique(image.flatten(), return_counts=True)
    occurrences = dict(zip(unique_nums, counts))

    kidney_intensities = (mask * image).flatten()

    unique_nums_kidney, counts_kidney = np.unique(kidney_intensities, return_counts=True)
    occurrences_kidney = dict(zip(unique_nums_kidney, counts_kidney))

    new_df_intensities = {**df_intensities, **occurrences}
    new_df_intensities_kidney = {**df_intensities_kidney, **occurrences_kidney}

    for key, value in df_intensities.items():
        if key in occurrences:
            new_df_intensities[key] = value + occurrences[key]

    for key, value in df_intensities_kidney.items():
        if key in occurrences_kidney:
            new_df_intensities_kidney[key] = value + occurrences_kidney[key]
    
    df_intensities = new_df_intensities
    df_intensities_kidney = new_df_intensities_kidney

#df.to_csv('reports/figures/Data_statistics/hepaticDataExploration.csv', index=False)

df.to_csv('reports/figures/Data_statistics/renalDataExploration.csv', index=False)
import pickle


# Save the dictionary to a file
with open('reports/figures/Data_statistics/renalDataExploration.pkl', 'wb') as f:
    pickle.dump(df_intensities, f)

with open('reports/figures/Data_statistics/mask_renalDataExploration.pkl', 'wb') as f:
    pickle.dump(df_intensities_kidney, f)

