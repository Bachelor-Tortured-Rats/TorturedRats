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
hepatic_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'

# get the names of all files in the directory
file_names = [f for f in os.listdir(hepatic_path + "imagesTr/") if not f.startswith('.')]


import pandas as pd
# for loop to collect data and add to dataframe
df = pd.DataFrame(columns=["filename", "dim0", "dim1", "dim2", "pixdim0", "pixdim1", "pixdim2"])



for fname in file_names:

    break ### REMOVE ###

    label_path = hepatic_path + ["labelsTr/", "imagesTr/"][1] + fname
    load_dict = load_transform({"image": label_path, "label": label_path})

    
    image = load_dict["image"]

    pixdims = load_dict['image_meta_dict']['pixdim'][1:4]
    shapes = image.shape

    df = df.append({"filename": fname, "dim0": shapes[0], "dim1": shapes[1], "dim2": shapes[2], "pixdim0": pixdims[0], "pixdim1": pixdims[1], "pixdim2": pixdims[2]}, ignore_index=True)


#df.to_csv('reports/figures/Data_statistics/hepaticDataExploration.csv', index=False)


min_vox = float("inf")
max_vox = -float("inf")
df = pd.read_csv('reports/figures/Data_statistics/hepaticDataExploration.csv')

for index, row in df.iterrows():

    fname = row["filename"]
    print(index)
    
    load_dict = load_transform({"image": hepatic_path + "imagesTr/" + fname , "label": hepatic_path + "labelsTr/" + fname})
    
    image = load_dict["image"]

    print(len(str(list(np.histogram(image.flatten(), bins=np.arange(-1024, 4545))[0]))))
    break

    df.loc[index, "img_hist"] = str(list(np.histogram(image.flatten(), bins=np.arange(-1024, 4545))[0]))

    label = load_dict["label"]

    label = label * image

    df.loc[index, "label_hist"] = str(list(np.histogram(label.flatten(), bins=np.arange(-1024, 4545))[0]))

    if np.min(image) < min_vox:
        min_vox = np.min(image)
    
    if np.max(image) > max_vox:
        max_vox = np.max(image)


print("Min: ", min_vox)
print("Max: ", max_vox)
df.to_csv('reports/figures/Data_statistics/hepaticDataExploration.csv', index=False)

