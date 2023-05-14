import torch
from aesthetic_tensor.core import AestheticTensor


def aesthetify():
    prop: AestheticTensor = property(
        lambda self: AestheticTensor(self),
    )
    torch.Tensor.ae: AestheticTensor = prop
    torch.Tensor.æ: AestheticTensor = prop
