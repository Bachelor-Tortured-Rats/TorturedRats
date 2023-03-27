import monai
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform, RandomizableTransform

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
        self.label_key = label_key
        self.rand = np.zeros((1,3))
        self.loc = {
            0: (0, 0),
            1: (0, 1),
            2: (0, 2),
            3: (1, 0),
            4: (1, 1),
            5: (1, 2),
            6: (2, 0),
            7: (2, 1),
            8: (2, 2)
        }
        super().__init__(keys)

    def randomize(self):
        super().randomize(None)
        if not self._do_transform:
            return None
        rands = self.R.randint(0, 7)
        if rands >= 4:
            rands += 1
        self.rand[0] = rands
        self.rand[1] = (self.loc[rands])[0]*16
        self.rand[2] = (self.loc[rands])[1]*16


    def __call__(self, data, randomize: bool = True):
        d = dict(data)
        # random number between 0 and 8 except for 4
        # rand = self.R.randint(0, 9, size=3)
        if randomize:
            self.randomize()
        
        if not self._do_transform:
            return None

        for key in self.key_iterator(d):
            #center[key] = CenterSpacialCrop(d[key], roi_size=(16, 16)) ## if we assume (48, 48)
            # gets center patch
            center = d[key][16:32, 16:32]
            # gets other patch
            other = d[key][self.rand[1]:self.rand[1]+16, self.rand[2]:self.rand[2]+16]

            d[key] = (center, other)
            d[self.label_key] = self.rand[0] # position
        return d