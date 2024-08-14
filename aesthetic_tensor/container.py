from ansi2html import Ansi2HTMLConverter
from PIL import Image
import numpy as np

from aesthetic_tensor.gif import GIF


class AestheticContainer:
    def __init__(self, aesthetic_tensor):
        self.container = [t for t in aesthetic_tensor]

    def _repr_html_(self):
        if not hasattr(self.container[0], "_repr_html_"):
            conv = Ansi2HTMLConverter()
            return conv.convert(repr(self))

        content = ""
        for item in self.container:
            item_repr = item._repr_html_()
            content += f"<div style='margin: 1px'>{item_repr}</div>"

        return f"""
            <div style="display: flex;flex-wrap: wrap;">
                {content}
            </div>
        """

    @property
    def UN(self):
        from aesthetic_tensor.tensor import AestheticTensor

        return AestheticTensor(np.stack(self.map(lambda t: t.raw).raw))

    def map(self, mapper):
        return AestheticContainer([mapper(t) for t in self.container])

    def loc(self, idx):
        return self.container[idx]

    def __repr__(self):
        return f"[{len(self.container)}](~" + repr(self.container[0]) + ")"

    def __getitem__(self, key):
        return AestheticContainer([t.__getitem__(key) for t in self.container])

    def __getattr__(self, key):
        return AestheticContainer([getattr(t, key) for t in self.container])

    def __call__(self, *args, **kwds):
        return AestheticContainer([t(*args, **kwds) for t in self.container])

    @property
    def gif(self):
        from aesthetic_tensor.tensor import AestheticTensor

        if not hasattr(self.container[0], "gif"):
            pils = [e.pil for e in self.container]
            return GIF(pils)

        return AestheticContainer([getattr(t, "gif") for t in self.container])

    @property
    def raw(self):
        return self.container
