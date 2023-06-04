import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.plot_functions import plot_three_slices, displaySlice, generate_dashed_lines
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
import cv2

load_transform = Compose([LoadImaged(keys=["image", "label"])])

# specify the path to the NIfTI file
data_dir = '/dtu/3d-imaging-center/projects/2020_QIM_22_rat_kidney/analysis/study_diabetic'

rats = [21, 22, 24,25,28,33,36,37,38,43,47,48,51,52,54,57]

train_images = [f'{data_dir}/aligned/rat{i}_aligned_rigid.nii' for i in rats]
train_masks = [f'{data_dir}/maskKidney/rat{i}_kidneyMaskProc.nii.gz' for i in rats]

i = 7
#img = nib.load('/dtu/3d-imaging-center/projects/2020_QIM_22_rat_kidney/analysis/analysis_rat37/rat37_reorient.nii.gz')
#lab_true = nib.load('/dtu/3d-imaging-center/projects/2020_QIM_22_rat_kidney/analysis/analysis_rat37/vessel_zoom_ground_truth-ish_rat37.nii.gz')
#load_dict = load_transform({"image": train_images[i], "label": train_masks[i]})
rat37 = '/dtu/3d-imaging-center/projects/2020_QIM_22_rat_kidney/analysis/analysis_rat37/rat37_reorient.nii.gz'
load_dict = load_transform({"image": rat37, "label": '/dtu/3d-imaging-center/projects/2020_QIM_22_rat_kidney/analysis/analysis_rat37/vessel_zoom_ground_truth-ish_rat37.nii.gz'})

CT_image = load_dict["image"]
mask  = load_dict["label"]

#CT_image = CT_image * mask

#CT_image = np.rot90(CT_image, k=1, axes=(0, 1))
#mask = np.rot90(mask, k=1, axes=(0, 1))

dim_depths = [500, 500, 500]


images = plot_three_slices(CT_image,  dim_depths, mask, alpha = 0.6, colors = [], thickness=2, dash_space=8)

"""
alpha = 0.5
img1 = images[1].astype(np.uint8)
img1_rectangle = img1.copy()

cv2.rectangle(img1_rectangle, pt1, pt2, (0, 0, 255), 3)

images[1] = img1 * (1-alpha) +  img1_rectangle * alpha
"""






fig = plt.figure(figsize=(6,4))
fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)

X = [ (1,3,1), (1,3,2), (1,3,3)]
titles = ['Sagittal plane', 'Coronal plane', 'Transverse plane']

for i, (nrows, ncols, plot_number) in enumerate(X):
    sub = fig.add_subplot(nrows, ncols, plot_number)
    sub.imshow(images[i])
    sub.set_title(titles[i])
    sub.set_xticks([])
    sub.set_yticks([])

plt.savefig("reports/figures/02_Data/raw_data_renal_microvessels.png")

plt.show()

exit()
with open('reports/figures/Data_statistics/renalDataExploration.pkl', 'rb') as f:
    d1 = pickle.load(f)

with open('reports/figures/Data_statistics/mask_renalDataExploration.pkl', 'rb') as f:
    d2 = pickle.load(f)

max_int = max(list(d1.keys()))

min_int = min(list(d1.keys()))

print(min_int, "  ", max_int)

all_values = np.arange(min_int, max_int + 1)

img_voxels = [d1[float(i + min_int)] if  float(i + min_int) in d1 else 0 for i in range(len(all_values))]
mask_voxels = [d2[float(i + min_int)] if  float(i + min_int) in d2 else 0 for i in range(len(all_values))]

img_hist = []
mask_hist = []
bin_edges = []
# Iterate over the array in groups of 10
jump = 1
for i in range(0, len(all_values), jump):
    # Sum the elements in the current group
    img_sum = sum(img_voxels[i:i+jump])
    mask_sum = sum(mask_voxels[i:i+jump])

    # Append the group sum to the result array
    img_hist.append(img_sum)
    mask_hist.append(mask_sum)

    bin_edges.append(all_values[i])

bin_edges.append(all_values[-1])
#print(all(np.isfinite(img_hist)))
#print(all(np.isfinite(mask_hist)))
#print(all(np.isfinite(bin_edges)))
print(max_int)
print(min_int)
print(len(bin_edges))

#plt.hist(img_hist[1:], bins=bin_edges[1:], color="red", density=True, alpha=0.5)
#plt.hist(mask_hist[1:], bins=bin_edges[1:], color="blue", density=True, alpha=0.5)
plt.bar(bin_edges[1:-1], img_hist[1:]/sum(img_hist[1:]), color="red", alpha=0.5)
plt.bar(bin_edges[1:-1], mask_hist[1:]/sum(mask_hist[1:]), color="blue",alpha=0.5)

plt.show()