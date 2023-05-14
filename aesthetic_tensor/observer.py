from IPython.display import display, clear_output
import ipywidgets as ipw


class AestheticObserver:
    def __init__(self, value, commands=[]) -> None:
        self.value = value
        self.commands = commands
        self.out = ipw.Output()

    def __repr__(self):
        commands = [n for n, _ in self.commands]
        return f"AestheticObserver<{len(commands)}-commands>({type(self.raw)})"

    def new_command(self, name, command):
        return AestheticObserver(self.value, self.commands + [(name, command)])

    def __getitem__(self, key):
        return self.new_command("getitem", lambda v: v.__getitem__(key))

    def __getattr__(self, key):
        if key == "_ipython_canary_method_should_not_exist_":
            raise AttributeError(f"object has no attribute '{key}'")

        return self.new_command("getattr", lambda v: getattr(v, key))

    def __call__(self, *args, **kwds):
        return self.new_command("call", lambda v: v(*args, **kwds))

    def _ipython_display_(self):
        self.render()
        display(self.out)

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
        self.render()

    @property
    def raw(self):
        v = self.value
        for n, c in self.commands:
            v = c(v)

        return v
