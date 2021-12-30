from dataclasses import dataclass
from typing import List, Tuple

import cv2
import easyocr
import numpy as np
from PIL import Image


@dataclass
class BBox:
    left: int
    right: int
    top: int
    bottom: int


class GraphScale:
    def __init__(self, pixel_pos: List[int], values: List[float]):
        """Constructs a GraphScale object, calibrated on 2 pixel positions and 2 corresponding values."""
        assert len(pixel_pos) == 2
        assert len(values) == 2
        self.c1 = (values[1] - values[0]) / (pixel_pos[1] - pixel_pos[0])
        self.c0 = values[0] - pixel_pos[0] * self.c1

    def pixel_to_value(self, pixel_pos: int) -> float:
        return self.c0 + (self.c1 * pixel_pos)

    def value_to_pixel(self, value: float) -> int:
        return round((value - self.c0) / self.c1)


class Graph:

    # --- constructor -------------------------------------
    def __init__(self, filename: str):
        self._pil_image = Image.open(filename).convert(mode="RGB")  # type: Image.Image

    # --- debugging ---------------------------------------
    def show(self):
        self._pil_image.show()

    # --- properties --------------------------------------
    @property
    def width(self) -> int:
        return self._pil_image.width

    @property
    def height(self) -> int:
        return self._pil_image.height

    def as_array(self) -> np.ndarray:
        """Returns image as (height, width, depth=3)-sized np array (dtype=unit8) representing 3 RGB channels."""
        return np.asarray(self._pil_image)

    # --- data extraction ---------------------------------
    def _ocr(self) -> List[Tuple[BBox, str, float]]:
        """Returns a list of (bounding_box, text, confidence)-tuples as found with EasyOCR."""

        # OCR using easyocr
        reader = easyocr.Reader(["en"])
        raw_result = reader.readtext(image=self.as_array())  # type: List[Tuple[List[List[int]], str, float]]

        # convert bounding box coordinate lists to BBox objects
        result = []
        for coord_list, text, confidence in raw_result:
            x_values = [coord[0] for coord in coord_list]
            y_values = [coord[1] for coord in coord_list]
            result.append((BBox(min(x_values), max(x_values), min(y_values), max(y_values)), text, confidence))

        return result

    def _detect_hor_lines(self, clr: Tuple[int, int, int], min_frac: float, clr_tol: float = 0.01) -> List[int]:
        pixels = self._detect_color(clr, clr_tol)
        min_pixel_count = int(self.width * min_frac)
        row_has_line = pixels.sum(axis=1) >= min_pixel_count
        return [i for i, j in enumerate(row_has_line) if j]

    def _detect_vert_lines(self, clr: Tuple[int, int, int], min_frac: float, clr_tol: float = 0.01) -> List[int]:
        pixels = self._detect_color(clr, clr_tol)
        min_pixel_count = int(self.height * min_frac)
        col_has_line = pixels.sum(axis=0) >= min_pixel_count
        return [i for i, j in enumerate(col_has_line) if j]

    # --- internal ----------------------------------------
    def _detect_color(self, clr: Tuple[int, int, int], clr_tol: float) -> np.ndarray:
        """Returns numpy array of size (height, width) with False or True (True where color occurs)"""
        clr_tol_255 = int(255 * clr_tol)

        clr_min = np.array([max(0, x - clr_tol_255) for x in clr], dtype="uint8")
        clr_max = np.array([min(255, x + clr_tol_255) for x in clr], dtype="uint8")

        pixels = cv2.inRange(self.as_array(), clr_min, clr_max)

        return pixels == 255
