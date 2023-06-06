import functools
from io import BytesIO

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision
from IPython.display import Image as ipy_Image
from PIL import Image

from aesthetic_tensor.broadcaster import hook
from aesthetic_tensor.container import AestheticContainer
from aesthetic_tensor.observer import AestheticObserver
from aesthetic_tensor.utils import patch_callable
from argparse import Namespace
import base64


def make_red(v):
    return f"\x1b[31m{v}\x1b[0m"


def make_bold(v):
    return f"\x1B[1m{v}\x1b[0m"


def fig_to_pil(fig):
    buf = BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    pil = Image.open(buf)
    # fig.canvas.draw()
    # pil = Image.frombytes(
    #     "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    # )
    return pil


class ImageWrapper:
    @staticmethod
    def from_fig(fig):
        return ImageWrapper(fig_to_pil(fig))

    def __init__(self, pil):
        self.pil = pil

    def _repr_html_(self):
        buffered = BytesIO()
        self.pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf8")
        return f"""<img src="data:image/png;base64, {img_str}" alt="img"/>"""


class MatplotlibMixin:
    def hist(self, **kwargs):
        flat = self.np.reshape(-1)
        fig, ax = plt.subplots(1, 1, **{"dpi": 110, "figsize": (3.5, 3), **kwargs})
        plt.tight_layout()
        sns.histplot(flat, bins=30, ax=ax)
        plt.close()
        return ImageWrapper.from_fig(fig)

    def plot(self, **kwargs):
        flat = self.np.reshape(-1)
        fig, ax = plt.subplots(1, 1, **{"dpi": 110, "figsize": (3.5, 3), **kwargs})
        sns.lineplot(flat, ax=ax)
        plt.tight_layout()
        plt.close()
        return ImageWrapper.from_fig(fig)

    def imshow(self, cmap="viridis", **kwargs):
        fig, ax = plt.subplots(1, 1, **{"dpi": 110, "figsize": (3.5, 3), **kwargs})
        ax.imshow(self.np, cmap=cmap)
        plt.tight_layout()
        plt.close()
        return ImageWrapper.from_fig(fig)

    @property
    def displot(self):
        flat = self.np.reshape(-1)
        info = self.info
        # fig, ax = plt.subplots(1, 1, **{"dpi": 110, "figsize": (3.5, 3)})
        fig = sns.displot(flat, kde=True, height=2.5, aspect=2)
        ax = fig.facet_axis(0, 0)
        mi_y, ma_y = ax.get_ylim()
        ax.text(info.mean, ma_y, "μ", rotation=0)
        ax.axvline(info.mean, c="k", ls="-", lw=1)
        plt.tight_layout()
        plt.close()
        return ImageWrapper.from_fig(fig)

    def cmap(self, cm="viridis", dim=-1):
        cmap = mpl.cm.get_cmap(cm)
        t = torch.tensor(cmap(self.np))
        dims = list(range(t.ndim))
        dims[dim], dims[-1] = dims[-1], dims[dim]
        t = t.permute(dims)
        return AestheticTensor(t).uint8


class AestheticTensor(MatplotlibMixin):
    def __init__(self, target):
        self.target: torch.Tensor = target
        functools.update_wrapper(self, torch.Tensor)

    def __getitem__(self, key):
        return AestheticTensor(self.target.__getitem__(key))

    def __getattr__(self, key):
        obj = getattr(self.target, key)
        if callable(obj):
            return patch_callable(
                obj,
                condition=lambda res: type(res) is torch.Tensor,
                type_wrapper=AestheticTensor,
            )
        return obj

    def dim_shift(self, size):
        shape = self.target.shape
        ndim = len(shape)
        shift = [((d - size) % ndim) + 1 for d in range(ndim)]
        return AestheticTensor(self.target.permute(*shift))

    @property
    def hwc(self):
        return AestheticTensor(self.target.permute(1, 2, 0))

    @property
    def chw(self):
        return AestheticTensor(self.target.permute(2, 0, 1))

    @property
    def info(self):
        target = self.target.to(torch.float64)
        nan_mask = torch.isnan(target)
        neg_inf_mask = torch.isneginf(target)
        inf_mask = torch.isinf(target) * ~neg_inf_mask
        valid_mask = ~nan_mask & ~inf_mask & ~neg_inf_mask

        mi = target[valid_mask].min()
        ma = target[valid_mask].max()
        std = target[valid_mask].std()
        mean = target[valid_mask].mean()

        return Namespace(
            shape=tuple(target.shape),
            range=(mi.item(), ma.item()),
            mean=mean.item(),
            std=std.item(),
            num_nan=torch.sum(nan_mask).item(),
            num_inf=torch.sum(inf_mask).item(),
            num_neg_inf=torch.sum(neg_inf_mask).item(),
        )

    def __repr__(self):
        info = self.info
        mi, ma = info.range

        shape_str = make_bold(", ".join(str(d) for d in info.shape))
        nan_str = (
            (", " + make_bold(make_red(f"∃nan*{info.num_nan}")))
            if info.num_nan > 0
            else ""
        )
        inf_str = (
            (", " + make_bold(make_red(f"∃∞*{info.num_inf}")))
            if info.num_inf > 0
            else ""
        )
        neg_inf_str = (
            (", " + make_bold(make_red(f"∃-∞*{torch.sum(info.num_neg_inf)}")))
            if info.num_neg_inf > 0
            else ""
        )
        range_str = make_bold(f"{mi:0.3f}, {ma:0.3f}")
        _, type_str = str(self.target.dtype).split(".")
        type_str = make_bold(type_str)

        mu_str = make_bold(f"{info.mean:0.3f}")
        std_str = make_bold(f"{info.std:0.3f}")
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
        pil = Image.fromarray(self.np)
        return ImageWrapper(pil)

    @property
    def gif(self):
        aesthetic_self = self

        class GIF:
            def __init__(self) -> None:
                self.fps_val = 30

            def __call__(self, fps):
                self.fps_val = fps
                return self

            def _repr_html_(self):
                fp = BytesIO()
                pils = [w.pil for w in aesthetic_self.N.img.raw]
                pils[0].save(
                    fp,
                    format="gif",
                    save_all=True,
                    append_images=pils[1:],
                    duration=1000 / self.fps_val,
                    loop=0,
                )
                fp.seek(0)

                b64 = base64.b64encode(fp.read()).decode("ascii")
                return f"""<img src="data:image/gif;base64,{b64}" />"""

        return GIF()

    @property
    def img(self):
        if self.target.ndim == 2:
            return self.cmap().pil
        elif self.target.ndim == 3:
            target = AestheticTensor(self.target)
            if target.dtype != torch.uint8:
                target = self.uint8
            if target.size(0) == 3:  # is in chw mode
                return target.hwc.pil
            return target.pil
        raise Exception("Invalid shape for image")

    def zoom(self, scale=1):
        assert self.ndim in [2, 3], "n-dims should be 2 or 3"
        t = self.target
        ndim = t.ndim
        if ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        else:  # assumes t.ndim == 3
            t = t.unsqueeze(0)

        hwc_test = t.shape[-1] == 3 and t.shape[1] > 3
        if hwc_test:
            t = t.permute(0, 3, 1, 2)

        t = F.interpolate(t, scale_factor=scale)

        if hwc_test:
            t = t.permute(0, 2, 3, 1)

        return AestheticTensor(t[0, 0] if ndim == 2 else t[0])

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
    def N(self):
        return AestheticContainer(self)

    @property
    def live(self):
        return AestheticObserver(self)

    def hook(self, *args):
        broadcasters, handler = args[:-1], args[-1]
        return hook(*broadcasters, lambda *vals: handler(self, *vals))

    @property
    def raw(self):
        return self.target
