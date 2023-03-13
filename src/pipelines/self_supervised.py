#### Elements ####
from monai.transforms import RandGridPatch
import nibabel as nib
from src.visualization.plot_functions import displaySlice 

# Load an example scan
exampleImage = nib.load("/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/imagesTr/hepaticvessel_458.nii.gz")
image = exampleImage.get_fdata()



# preprocessing
## lave patches fra et input volumen
import numpy as np

def extract_patches(image, patch_size=(8,8,8)):
    """Extracts 3D patches of the given patch size from a 3D image"""
    patches = {}
    z_patches, y_patches, x_patches = image.shape[0] // patch_size[0], image.shape[1] // patch_size[1], image.shape[2] // patch_size[2]
    
    for z in range(z_patches):
        for y in range(y_patches):
            for x in range(x_patches):
                patch = image[z*patch_size[0]:(z+1)*patch_size[0], y*patch_size[1]:(y+1)*patch_size[1], x*patch_size[2]:(x+1)*patch_size[2]]
                patches[(z, y, x)] = patch
    
    return patches


patches = extract_patches(image, patch_size = ())
print(patches)

# Returns a 20x20x20 patch

# En encoder model
## Kan tage en patch af samme dim som ved inference og kommer ud med en latent kode




# En context encoder model
## Kan tage en række patches og returnere et sæt af latente koder for hver af dem


# loss function




