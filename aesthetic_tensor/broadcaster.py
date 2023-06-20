from functools import wraps

import ipywidgets as ipw
from IPython.display import display

from aesthetic_tensor.observer import AestheticObserver


class AestheticBroadcaster:
    def __init__(self, value=None):
        self.value = value
        self.on_change_handlers = []

    def on_change(self, on_change_handler):
        self.on_change_handlers.append(on_change_handler)

    def update(self, value):
        self.value = value
        self.notify()

    def notify(self):
        for handler in self.on_change_handlers:
            handler(self.value)


def hook(*args):
    broadcasters, observer_handler = args[:-1], args[-1]
    values = [b.value for b in broadcasters]
    observer = AestheticObserver(lambda: observer_handler(*values))()

    def handler(*a):
        values = [b.value for b in broadcasters]
        observer.update(lambda: observer_handler(*values))

    for b in broadcasters:
        b.on_change(handler)
    return observer


class IPW:
    def __init__(self) -> None:
        for key in dir(ipw):
            cls = getattr(ipw, key)
            try:
                if issubclass(cls, ipw.Widget):
                    setattr(self, key, self.map_class(cls))
            except TypeError:
                # TODO: Fix this ugly hack!
                # Reason: function objects cannot be called as first argument of issubclass
                pass

    def map_class(self, cls):
        class new_cls(AestheticBroadcaster):
            @wraps(cls)
            def __init__(self, *args, **kwargs):
                self.el = cls(*args, **kwargs)
                super().__init__(self.el.value)

                def update(e):
                    self.update(self.el.value)

                self.el.observe(update)

            def update(self, value):
                self.value = value
                self.el.value = value
                super().notify()

            def _ipython_display_(self):
                display(self.el)

        return new_cls


ipw = IPW()
