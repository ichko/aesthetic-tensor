import torch
from aesthetic_tensor.tensor import AestheticTensor
from aesthetic_tensor.broadcaster import AestheticBroadcaster, ipw, hook

V = AestheticBroadcaster


def __call__(self):
    print(self)


def aesthetify():
    prop: AestheticTensor = property(
        lambda self: AestheticTensor(self),
    )
    torch.Tensor.ae: AestheticTensor = prop
    torch.Tensor.æ: AestheticTensor = prop
