#### Elements ####
from monai.transforms import RandGridPatch
import nibabel as nib
from src.visualization.plot_functions import displaySlice 

# Load an example scan
exampleImage = nib.load("/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/imagesTr/hepaticvessel_458.nii.gz")
image = nib.get_fdata(exampleImage)

displaySlice(image, 0, 50)

# preprocessing
## lave patches fra et input volumen

RandGridPatch



# En encoder model
## Kan tage en patch af samme dim som ved inference og kommer ud med en latent kode




# En context encoder model
## Kan tage en række patches og returnere et sæt af latente koder for hver af dem


# loss function




