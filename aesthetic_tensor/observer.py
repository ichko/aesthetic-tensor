import ipywidgets as ipw
from IPython.display import clear_output, display


class AestheticObserver:
    def __init__(self, value, commands=[]):
        self.value = value
        self.commands = commands
        self.out = ipw.Output()
        self.displayed = False

    def __repr__(self):
        # commands = [n for n, _ in self.commands]
        return f"*{repr(self.raw)}"

    def new_command(self, name, command):
        return AestheticObserver(self.value, self.commands + [(name, command)])

    def __getitem__(self, key):
        return self.new_command("getitem", lambda v: v.__getitem__(key))

    def __getattr__(self, key):
        """
        Attention:
        This is done so that the method `_ipython_display_` be called.
        Jupyter checks for this missing method to check if __getattr__
        has been defined. If so, a normal __repr__ is called.
        """
        if key == "_ipython_canary_method_should_not_exist_":
            raise AttributeError(f"object has no attribute '{key}'")

        return self.new_command("getattr", lambda v: getattr(v, key))

    def __call__(self, *args, **kwds):
        return self.new_command("call", lambda v: v(*args, **kwds))

    def display(self):
        self.displayed = True
        self.render()
        display(self.out)

    def _ipython_display_(self):
        self.display()

    def render(self):
        r = self.raw
        with self.out:
            """
            Attention:
            This fixes an issue display updates
            <https://discourse.jupyter.org/t/jupyter-notebook-zmq-message-arrived-on-closed-channel-error/17869/27>
            """
            clear_output(wait=True)
            display(r)

    def update(self, value):
        self.value = value
        if self.displayed:
            self.render()

    @property
    def raw(self):
        v = self.value
        for n, c in self.commands:
            v = c(v)
        return v
