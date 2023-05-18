from PIL import Image
import torch


class AestheticContainer:
    def __init__(self, aesthetic_tensor):
        self.container = [t for t in aesthetic_tensor]

    def nb(self, ncol=-1):
        import ipywidgets as ipw
        from IPython.display import display

        vertical = [[]]
        for i, t in enumerate(self.container):
            o = ipw.Output()
            if type(t) is Image.Image:
                with o:
                    display(t)
            else:
                o.append_display_data(t)

            vertical[-1].append(o)
            if i % ncol == ncol - 1:
                vertical[-1] = ipw.HBox(vertical[-1])
                vertical.append([])

        if len(vertical[-1]) > 0:
            vertical[-1] = ipw.HBox(vertical[-1])
        else:
            vertical.pop()

        return ipw.VBox(vertical)

    @property
    def N(self):
        return torch.stack(self.map(lambda t: t.raw).raw).ae

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
    def raw(self):
        return self.container
