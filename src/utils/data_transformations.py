import monai
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform, RandomizableTransform
import pdb

class Addd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
    ) -> None:
        
        self.keys = keys
        self.source_key = source_key
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = d[key] + d[self.source_key]
        return d
    
class selectPatchesd(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys: KeysCollection
    ) -> None:
        
        self.keys = keys
        self.label_key = 'label'
        self.rand = np.zeros(4,dtype=int)
        self.loc = {
            0: (0, 0, 0),
            1: (0, 0, 1),
            2: (0, 0, 2),
            3: (0, 1, 0),
            4: (0, 1, 1),
            5: (0, 1, 2),
            6: (0, 2, 0),            
            7: (0, 2, 1),            
            8: (0, 2, 2),
            9: (1, 0, 0),
            10: (1, 0, 1),
            11: (1, 0, 2),
            12: (1, 1, 0),
            13: (1, 1, 1), # center patch
            14: (1, 1, 2),
            15: (1, 2, 0),
            16: (1, 2, 1),
            17: (1, 2, 2),
            18: (2, 0, 0),
            19: (2, 0, 1),
            20: (2, 0, 2),
            21: (2, 1, 0),
            22: (2, 1, 1),
            23: (2, 1, 2),
            24: (2, 2, 0),
            25: (2, 2, 1),
            26: (2, 2, 2)
        }
        super().__init__()

    def randomize(self):
        super().randomize(None)
        if not self._do_transform:
            return None
        rands = self.R.randint(0, 26)
        if rands >= 13:
            rands += 1
        
        self.rand[0] = rands
        self.rand[1] = (self.loc[rands])[0]*16
        self.rand[2] = (self.loc[rands])[1]*16
        self.rand[3] = (self.loc[rands])[2]*8


    def __call__(self, data, randomize: bool = True):
        d = dict(data) # keys: image

        # calculates a new position for the path
        if randomize:
            self.randomize()
        
        if not self._do_transform:
            return None

        for key in self.key_iterator(d):
            #center[key] = CenterSpacialCrop(d[key], roi_size=(16, 16)) ## if we assume (48, 48)
            # gets center patch
            #print(d[key].shape)
            
            center = d[key][0,16:32, 16:32, 8:16]
            # gets other patch
            other = d[key][0, self.rand[1]:self.rand[1]+16, self.rand[2]:self.rand[2]+16, self.rand[3]:self.rand[3]+8]

            # updates keys in dictionary with the new pathches and the new position
            #d[key+"_center"] = center
            #d[key+"_other"] = other
            d[key] = [center, other]
            #d[key] = other
            d[self.label_key] = self.rand[0] # position
            

        return d
    

class RandSelectPatchesd(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
    ) -> None:
        
        self.keys = keys
        self.label_key = 'label'
        self.rand = np.zeros(4,dtype=int)
        self.rand_off = np.zeros(3,dtype=int)
        self.loc = {
             0: (0-1, 0-1, 0-1),
             1: (0-1, 0-1, 1-1),
             2: (0-1, 0-1, 2-1),
             3: (0-1, 1-1, 0-1),
             4: (0-1, 1-1, 1-1),
             5: (0-1, 1-1, 2-1),
             6: (0-1, 2-1, 0-1),            
             7: (0-1, 2-1, 1-1),            
             8: (0-1, 2-1, 2-1),
             9: (1-1, 0-1, 0-1),
            10: (1-1, 0-1, 1-1),
            11: (1-1, 0-1, 2-1),
            12: (1-1, 1-1, 0-1),
            13: (1-1, 1-1, 1-1), # center patch
            14: (1-1, 1-1, 2-1),
            15: (1-1, 2-1, 0-1),
            16: (1-1, 2-1, 1-1),
            17: (1-1, 2-1, 2-1),
            18: (2-1, 0-1, 0-1),
            19: (2-1, 0-1, 1-1),
            20: (2-1, 0-1, 2-1),
            21: (2-1, 1-1, 0-1),
            22: (2-1, 1-1, 1-1),
            23: (2-1, 1-1, 2-1),
            24: (2-1, 2-1, 0-1),
            25: (2-1, 2-1, 1-1),
            26: (2-1, 2-1, 2-1)
        }
        super().__init__()

    # Chooses a random patch
    def randomize(self):
        super().randomize(None)
        if not self._do_transform:
            return None
        rands = self.R.randint(0, 26)
        if rands >= 13:
            rands += 1
        #54 ,54 30
        #96, 96, 96
        
        self.rand[0] = rands
        self.rand[1] = 19 + (self.loc[rands])[0]*16 # x start pos
        self.rand[2] = 19 + (self.loc[rands])[1]*16 # y start pos
        self.rand[3] = 11 + (self.loc[rands])[2]*8 # z start pos

        self.rand_off = [self.R.randint(1,3),self.R.randint(1,3),self.R.randint(1,3)]


    def __call__(self, data, randomize: bool = True):
        d = dict(data) # keys: image

        # Selects random relative location for patch
        if randomize:
            self.randomize()
        
        if not self._do_transform:
            return None

        # For each random crop returned by RandCropByPosNegLabel
        # we extract the center patch and the randomly chosen patch
        for key in self.key_iterator(d):
            
            # Extract center patch
            center = d[key][0,19:35, 19:35, 11:19]
            
            # get random offset
            randx = -self.rand_off[0] if self.rand[0] in [0,1,2,3,4,5,6,7,8] else self.rand_off[0]
            randy = -self.rand_off[1] if self.rand[0] in [0,1,2,9,10,11,18,19,20] else self.rand_off[1]
            randz = -self.rand_off[2] if self.rand[0] in [0,3,6,9,12,15,18,21,24] else self.rand_off[2]
            
            # randx = np.random.randint(1,2+1)
            # randy = np.random.randint(1,2+1)
            # randz = np.random.randint(1,2+1)
            
            # gets other patch
            other = d[key][0, self.rand[1]+randx:self.rand[1]+16+randx, self.rand[2]+randy:self.rand[2]+16+randy, self.rand[3]+randz:self.rand[3]+8+randz]

       
            # Insert elements into dictionary 
            d[key] = [center, other]
            d[self.label_key] = self.rand[0] # position
            

        return d

class RandSelectPatchesLarged(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
    ) -> None:
        self.keys = keys
        self.label_key = 'label'
        self.rand = np.zeros(4,dtype=int)
        self.rand_off = np.zeros(3,dtype=int)
        self.loc = {
             0: (0-1, 0-1, 0-1),
             1: (0-1, 0-1, 1-1),
             2: (0-1, 0-1, 2-1),
             3: (0-1, 1-1, 0-1),
             4: (0-1, 1-1, 1-1),
             5: (0-1, 1-1, 2-1),
             6: (0-1, 2-1, 0-1),            
             7: (0-1, 2-1, 1-1),            
             8: (0-1, 2-1, 2-1),
             9: (1-1, 0-1, 0-1),
            10: (1-1, 0-1, 1-1),
            11: (1-1, 0-1, 2-1),
            12: (1-1, 1-1, 0-1),
            13: (1-1, 1-1, 1-1), # center patch
            14: (1-1, 1-1, 2-1),
            15: (1-1, 2-1, 0-1),
            16: (1-1, 2-1, 1-1),
            17: (1-1, 2-1, 2-1),
            18: (2-1, 0-1, 0-1),
            19: (2-1, 0-1, 1-1),
            20: (2-1, 0-1, 2-1),
            21: (2-1, 1-1, 0-1),
            22: (2-1, 1-1, 1-1),
            23: (2-1, 1-1, 2-1),
            24: (2-1, 2-1, 0-1),
            25: (2-1, 2-1, 1-1),
            26: (2-1, 2-1, 2-1)
        }
        super().__init__()

    # Chooses a random patch
    def randomize(self):
        super().randomize(None)
        if not self._do_transform:
            return None
        rands = self.R.randint(0, 26)
        if rands >= 13:
            rands += 1
        
        self.rand[0] = rands
        self.rand[1] = 100 + (self.loc[rands])[0]*96 # x start pos
        self.rand[2] = 100 + (self.loc[rands])[1]*96 # y start pos
        self.rand[3] = 100 + (self.loc[rands])[2]*96 # z start pos

        self.rand_off = [self.R.randint(1,3),self.R.randint(1,3),self.R.randint(1,3)]


    def __call__(self, data, randomize: bool = True):
        d = dict(data) # keys: image

        # Selects random relative location for patch
        if randomize:
            self.randomize()
        
        if not self._do_transform:
            return None

        # For each random crop returned by RandCropByPosNegLabel
        # we extract the center patch and the randomly chosen patch
        for key in self.key_iterator(d):
            
            # Extract center patch
            center = d[key][0,100:196, 100:196, 100:196]
            
            # get random offset
            randx = -self.rand_off[0] if self.rand[0] in [0,1,2,3,4,5,6,7,8] else self.rand_off[0]
            randy = -self.rand_off[1] if self.rand[0] in [0,1,2,9,10,11,18,19,20] else self.rand_off[1]
            randz = -self.rand_off[2] if self.rand[0] in [0,3,6,9,12,15,18,21,24] else self.rand_off[2]

            # gets other patch
            other = d[key][0, self.rand[1]+randx:self.rand[1]+96+randx, self.rand[2]+randy:self.rand[2]+96+randy, self.rand[3]+randz:self.rand[3]+96+randz]

       
            # Insert elements into dictionary 
            d[key] = [center, other]
            d[self.label_key] = self.rand[0] # position
            

        return d

class image_checker(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
    ) -> None:
        
        self.keys = keys
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            #print(d[key].shape[3])
            if d[key].shape[3] <= 48:
                print(d['image_meta_dict']['filename_or_obj'])
                print("Image too small")
        return d

if __name__ == "__main__":
    import numpy as np
    from monai.transforms import (Compose)
    # Train transforms to use for self supervised learning
    # on the hepatic dataset
    transforms_3drpl = Compose(
        [
            RandSelectPatchesd(keys=["image"]) # This one makes a random offset from the middle
        ]
    )

    print("Creating image")

    # Create coordinate arrays for each dimension
    x = np.arange(54)
    y = np.arange(54)
    z = np.arange(30)

    # Create a 3D grid using meshgrid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Combine the coordinate arrays into tuples
    grid = np.stack((xx, yy, zz), axis=-1)
    
    # creae 3d image as numpy array with size 100x100x100 filled with values from 1 to 1000000
    img = grid.reshape((1,54, 54, 30,3))
    print(img[0,0,0,0,:])
    # convert last dimension of img to tuple
    
    # convert last dimension to touple
    img = transforms_3drpl({"image":img})

    center_path, other_path = img['image'][0], img['image'][1]

    # printer difference between center_path and other_path
    #get the central item from the center_path array

    print(center_path.shape)
    print(other_path.shape)

    #center_path[x,y,z]
    print(img['label'])

    # get max and min value for each axis of the center_path
    print("min :", np.min(center_path[:,:,:,0]), "max :",  np.max(center_path[:,:,:,0]))
    print("min :", np.min(center_path[:,:,:,1]), "max :", np.max(center_path[:,:,:,1]))
    print("min :", np.min(center_path[:,:,:,2]), "max :", np.max(center_path[:,:,:,2]))

    print("min :", np.min(other_path[:,:,:,0]), "max :", np.max(other_path[:,:,:,0]))
    print("min :", np.min(other_path[:,:,:,1]), "max :", np.max(other_path[:,:,:,1]))
    print("min :", np.min(other_path[:,:,:,2]), "max :", np.max(other_path[:,:,:,2]))
    
