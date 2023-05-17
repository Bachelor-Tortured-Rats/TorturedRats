###
### Original script from: Vedrana Andersen Dahl
###

# Load packages.
import numpy as np
import skimage.measure
import nibabel as nib
import volvizplotly as vvp

img = nib.load('/zhome/a2/4/155672/Desktop/Bachelor/Task08_HepaticVessel/imagesTr/hepaticvessel_002.nii.gz')
lab_true = nib.load('/zhome/a2/4/155672/Desktop/Bachelor/Task08_HepaticVessel/labelsTr/hepaticvessel_002.nii.gz')
lab_pred = nib.load('reports/save_prediction_mask/hepaticvessel_002/hepaticvessel_002_seg.nii.gz')


vol = np.array(img.get_fdata()).transpose((2, 1, 0))
seg_true = np.array(lab_true.get_fdata()).transpose((2, 1, 0))
seg_pred = np.array(lab_pred.get_fdata()).transpose((2, 1, 0))

# Cropping to xy-bounding box of labels
# bb = seg.any(axis=0)
# x = np.where(bb.any(axis=0))[0]
# y = np.where(bb.any(axis=1))[0]
# vol = vol[:, y[0]: y[-1], x[0]: x[-1]]
# seg = seg[:, y[0]: y[-1], x[0]: x[-1]]

# Use marching cubes to obtain the surface mesh 
verts1, faces1, _, _ = skimage.measure.marching_cubes(seg_true==1, 0.5)
verts2, faces2, _, _ = skimage.measure.marching_cubes(seg_pred==1, 0.5)


fig = vvp.volume_slicer(vol, ['mid', None, None], show=False)
vvp.show_mesh(verts1, faces1, fig=fig, show=False, surface_color='red')
vvp.show_mesh(verts2, faces2, fig=fig, surface_color='green')
