import torch
import torch.nn as nn
from Make_patches import *


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        """
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )
        """
    
    def forward(self, x):
        # Encode
        x = self.encoder(x)
        # Decode
        #x = self.decoder(x)
        return x


if __name__ == "__main__":
    # Load an example scan
    exampleImage = nib.load("/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/imagesTr/hepaticvessel_458.nii.gz")
    image = exampleImage.get_fdata()

    # Create an instance of the model
    autoencoder = Autoencoder()

    # Create some input data (e.g., a single-channel 3D image with dimensions 8x8x8)
    #x = torch.randn(1, 1, 8, 8, 8)
    x = extract_patches(image, patch_size=(20,20,20))
    x = torch.from_numpy(np.expand_dims(x[(15, 15, 5)], axis=(0,1))).double()

    # Pass the input through the model to obtain the reconstructed output
    output = autoencoder(x)

    # Compute the reconstruction loss (e.g., using mean squared error)
    #loss_fn = nn.MSELoss()
    #loss = loss_fn(output, x)

    #print(loss)
    print(output)
    print(output.shape)