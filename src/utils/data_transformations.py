import monai
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform

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