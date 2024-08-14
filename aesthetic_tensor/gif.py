import base64
from io import BytesIO


class GIF:
    def __init__(self, pils, fps=24) -> None:
        self.fps_val = fps
        self.pils = pils

    def __call__(self, fps):
        self.fps_val = fps
        return self

    def _repr_html_(self):
        fp = BytesIO()
        self.pils[0].save(
            fp,
            format="gif",
            save_all=True,
            append_images=self.pils[1:],
            duration=1000 // self.fps_val,
            loop=0,
        )
        fp.seek(0)

        b64 = base64.b64encode(fp.read()).decode("ascii")
        return f"""<img src="data:image/gif;base64,{b64}" />"""
