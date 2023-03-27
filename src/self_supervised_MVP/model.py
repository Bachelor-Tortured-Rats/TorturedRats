import torch
import torch.nn as nn
import pdb

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        return x

class BeefierEncoder(nn.Module):
    def __init__(self):
        super(BeefierEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        return x

class Pred_head(nn.Module):
    def __init__(self):
        super(Pred_head, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            #nn.Softmax(dim=1)
        )

    def forward(self, patches_A, patches_B):
        # Stack patches as one vector
        x = torch.cat((patches_A, patches_B), dim=1)
        return self.layers(x)

class Beefier_Pred_head(nn.Module):
    def __init__(self):
        super(Beefier_Pred_head, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            #nn.Softmax(dim=1)
        )

    def forward(self, patches_A, patches_B):
        # Stack patches as one vector
        x = torch.cat((patches_A, patches_B), dim=1)
        return self.layers(x)

class SelfSupervisedModel(nn.Module):
    def __init__(self, CNNModel, preHeadModel):
        super(SelfSupervisedModel, self).__init__()
        self.CNNModel = CNNModel
        self.preHeadModel = preHeadModel
        
    def forward(self, center_patch, offset_patch):
        center_patch = self.CNNModel(center_patch)
        offset_patch = self.CNNModel(offset_patch)

        return self.preHeadModel(center_patch, offset_patch)


if __name__ == "__main__":
    # Load an example scan
    exampleImage = nib.load("/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/imagesTr/hepaticvessel_458.nii.gz")
    image = exampleImage.get_fdata()[:,:,100]

    # Create an instance of the model
    encoder = Encoder()
    pred_head = Pred_head()

    # Create some input data (e.g., a single-channel 3D image with dimensions 8x8x8)
    #x = torch.randn(1, 1, 8, 8, 8)
    #x = extract_patches(image, patch_size=(20,20,20))
    x = torch.from_numpy(np.expand_dims(image, axis=(0,1))).float() #.double()

    print(x.shape)

    # Pass the input through the model to obtain the reconstructed output
    output = encoder(x)

    print(output.shape)
    
    prediction = pred_head(output)
    
    print(prediction)


    #print(loss)
    #print(output)
    #print(output.shape)
    #print("test for my push issue")