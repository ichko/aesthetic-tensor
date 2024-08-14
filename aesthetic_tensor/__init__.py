import numpy as np
from aesthetic_tensor.broadcaster import AestheticBroadcaster, hook, ipw
from aesthetic_tensor.tensor import AestheticTensor

V = AestheticBroadcaster


def ae(array: np.ndarray):
    return AestheticTensor(array)


def monkey_patch_torch():
    import torch

    prop: AestheticTensor = property(
        lambda self: AestheticTensor(self.detach().cpu().numpy()),
    )
    torch.Tensor.ae = prop
    torch.Tensor.æ = prop
