import torch

from aesthetic_tensor.broadcaster import AestheticBroadcaster, hook, ipw
from aesthetic_tensor.tensor import AestheticTensor

V = AestheticBroadcaster


def aesthetify():
    prop: AestheticTensor = property(
        lambda self: AestheticTensor(self),
    )
    torch.Tensor.ae: AestheticTensor = prop
    torch.Tensor.æ: AestheticTensor = prop
