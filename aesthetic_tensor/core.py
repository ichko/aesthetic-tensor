import functools

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image


def make_red(v):
    return f"\x1b[31m{v}\x1b[0m"


def make_bold(v):
    return f"\x1B[1m{v}\x1b[0m"


def patch_callable(callable):
    @functools.wraps(callable)
    def new_callable(*args, **kwargs):
        result = callable(*args, **kwargs)
        if type(result) is torch.Tensor:
            return AestheticTensor(result)
        return result

    return new_callable


class AestheticTensor:
    def __init__(self, target):
        self.target = target
        functools.update_wrapper(self, torch.Tensor)

    def __getitem__(self, key):
        return AestheticTensor(self.target.__getitem__(key))

    def __getattr__(self, key):
        obj = getattr(self.target, key)
        if callable(obj):
            return patch_callable(obj)
        elif type(obj) is torch.Tensor:
            return AestheticTensor(obj)
        return obj

    def __call__(self, semantic_dims):
        pass

    def dim_shift(self, size):
        shape = self.target.shape
        ndim = len(shape) - 1  # no batch
        shift = [((d - size) % ndim) + 1 for d in range(ndim)]
        return AestheticTensor(self.target.permute(0, *shift))

    def cmap(self, cm="viridis"):
        cmap = mpl.cm.get_cmap(cm)
        t = torch.tensor(cmap(self.normal.np))
        return AestheticTensor(t).dim_shift(1).uint8

    @property
    def hist(self):
        flat = self.np.reshape(-1)

        fig, ax = plt.subplots(1, 1, dpi=128, figsize=(4, 2))
        sns.histplot(flat, ax=ax)
        plt.close()
        return fig

    def __repr__(self):
        target = self.target.to(torch.float64)
        nan_mask = torch.isnan(target)
        neg_inf_mask = torch.isneginf(target)
        inf_mask = torch.isinf(target) * ~neg_inf_mask
        has_nan = torch.any(nan_mask)
        has_inf = torch.any(inf_mask)
        has_neg_inf = torch.any(neg_inf_mask)
        valid_mask = ~nan_mask & ~inf_mask & ~neg_inf_mask

        mi = target[valid_mask].min()
        ma = target[valid_mask].max()
        shape_str = make_bold(", ".join(str(d) for d in target.shape))
        std = target[valid_mask].std()
        mean = target[valid_mask].mean()
        nan_str = (
            (", " + make_bold(make_red(f"∃nan*{torch.sum(nan_mask)}")))
            if has_nan
            else ""
        )
        inf_str = (
            (", " + make_bold(make_red(f"∃∞*{torch.sum(inf_mask)}"))) if has_inf else ""
        )
        neg_inf_str = (
            (", " + make_bold(make_red(f"∃-∞*{torch.sum(neg_inf_mask)}")))
            if has_neg_inf
            else ""
        )
        range_str = make_bold(f"{mi:0.5f}, {ma:0.5f}")
        _, type_str = str(self.target.dtype).split(".")
        type_str = make_bold(type_str)

        mu_str = make_bold(f"{mean:0.5f}")
        std_str = make_bold(f"{std:0.5f}")
        mean_std_str = f"μ={mu_str}, σ={std_str}"

        return f"{type_str}<{shape_str}>∈[{range_str}] | {mean_std_str}{nan_str}{inf_str}{neg_inf_str}"

    @property
    def np(self):
        return self.target.detach().cpu().numpy()

    def grid(self, nrow=8, pad=2):
        out = torchvision.utils.make_grid(
            self.target, nrow=nrow, padding=pad
        ).unsqueeze(0)
        return AestheticTensor(out)

    @property
    def uint8(self):
        return AestheticTensor((self.target * 255).to(torch.uint8))

    @property
    def pil(self):
        return [Image.fromarray(im) for im in self.dim_shift(-1).np]

    def zoom(self, scale=1):
        return AestheticTensor(F.interpolate(self.target, scale_factor=scale))

    @property
    def normal(self) -> "AestheticTensor":
        t = self.target
        shape = t.shape
        bs = shape[0]
        t_arr = t.reshape(bs, -1)
        mi = t_arr.min(dim=1).values.view(bs, *([1] * (len(shape) - 1)))
        ma = t_arr.max(dim=1).values.view(bs, *([1] * (len(shape) - 1)))
        return AestheticTensor((t - mi) / (ma - mi))

    @property
    def raw(self):
        return self.target


class AestheticCollection(AestheticTensor):
    def __init__(self, ae_tensor, semantic_dims):
        self.
