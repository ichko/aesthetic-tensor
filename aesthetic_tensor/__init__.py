import numpy as np
from aesthetic_tensor.broadcaster import AestheticBroadcaster, hook, ipw
from aesthetic_tensor.tensor import AestheticTensor

V = AestheticBroadcaster

def ae(array: np.ndarray):
    return AestheticTensor(array)

def monkey_patch_torch():
    prop: AestheticTensor = property(
        lambda self: AestheticTensor(self),
    )
    torch.Tensor.ae: AestheticTensor = prop
    torch.Tensor.æ: AestheticTensor = prop
