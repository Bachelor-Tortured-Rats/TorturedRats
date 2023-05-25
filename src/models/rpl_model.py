import torch
import torch.nn as nn
import pdb

class RelativePathLocationModel(nn.Module):
    def __init__(self, UnetEncoder, preHeadModel):
        super(RelativePathLocationModel, self).__init__()
        self.UnetEncoder = UnetEncoder
        self.preHeadModel = preHeadModel
        
    def forward(self, patches):
        center_patch, offset_patch = patches[0], patches[1]
        center_patch = self.UnetEncoder(center_patch)
        center_patch = torch.flatten(center_patch,start_dim=1)

        offset_patch = self.UnetEncoder(offset_patch)
        offset_patch = torch.flatten(offset_patch,start_dim=1)
        
        return self.preHeadModel(center_patch, offset_patch)
    

class RelativePathLocationModelHead(nn.Module):
    def __init__(self, input_dim=1024):
        super(RelativePathLocationModelHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 27),
        )

    def forward(self, patches_A, patches_B):
        # Stack patches as one vector
        x = torch.cat((patches_A, patches_B), dim=1)
        return self.layers(x)