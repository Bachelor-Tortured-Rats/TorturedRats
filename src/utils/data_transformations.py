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
        self.rand = np.zeros(4,dtype=np.int)
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
        self.rand[3] = (self.loc[rands])[2]*16


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
            
            center = d[key][0,16:32, 16:32, 16:32]
            # gets other patch
            other = d[key][0, self.rand[1]:self.rand[1]+16, self.rand[2]:self.rand[2]+16, self.rand[3]:self.rand[3]+16]

            # updates keys in dictionary with the new pathches and the new position
            d[key] = (center, other)
            d[self.label_key] = self.rand[0] # position
            

        return d