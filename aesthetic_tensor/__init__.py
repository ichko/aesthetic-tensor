import torch
from aesthetic_tensor.core import AestheticTensor


def aesthetify():
    torch.Tensor.æ = torch.Tensor.ae = property(
        lambda self: AestheticTensor(self),
    )
