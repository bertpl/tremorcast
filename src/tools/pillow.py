from typing import Any, Tuple

import numpy as np
from PIL import ImageDraw, ImageFont

from ..base.data_extraction.graph import BBox


class ImageDrawPlus(ImageDraw.ImageDraw):

    _DEFAULT_TTF_FILE = r"_fonts/open_sans_bold.ttf"
    _LINE_STYLES = {"-": None, "--": (10, [(0, 5)]), "..": (2, [(0, 0)]), ".-": (11, [(0, 5), (8, 8)])}

    def rectangle_from_bbox(self, bbox: BBox, fill: Any = None, outline: Any = None, width: int = 1, offset: int = 0):
        self.rectangle(
            xy=[(bbox.left - offset, bbox.top - offset), (bbox.right + offset, bbox.bottom + offset)],
            fill=fill,
            outline=outline,
            width=width,
        )

    def default_text(self, xy, text, size, *, fill: Any = None):
        fnt = ImageFont.truetype(self._DEFAULT_TTF_FILE, size=size)
        self.text(xy, text, font=fnt, fill=fill)

    def line_with_style(self, xy_from: Tuple[float, float], xy_to: Tuple[float, float], fill=None, style: str = "-"):
        """style = one of ("-", "--", "..", ".-")"""
        pattern_len, pattern = self._LINE_STYLES[style]

        def interp_xy(c: float) -> Tuple[int, int]:
            return round(x_from + c * (x_to - x_from)), round(y_from + c * (y_to - y_from))

        x_from, y_from = xy_from
        x_to, y_to = xy_to

        n = max(abs(x_from - x_to), abs(y_from - y_to))
        n_patterns = round(n / pattern_len) + 1
        sub_lines = [
            ((i * pattern_len) + i_from, (i * pattern_len) + i_to)
            for i_from, i_to in pattern
            for i in range(n_patterns)
        ]
        for i_from, i_to in sub_lines:
            if i_from <= n:
                xy = [interp_xy(i_from / n), interp_xy(min(i_to / n, 1))]
                self.line(xy, fill)
