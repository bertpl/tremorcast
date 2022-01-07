import datetime
from functools import cached_property
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from src.base.data_extraction import BBox, Graph, GraphScale
from src.base.time_series import TimeSeries
from src.tools.datetime import float_to_ts, ts_to_float
from src.tools.pillow import ImageDrawPlus

from .custom_time_series import MinMidMaxTimeSeries
from .vedur import VedurColors
from .vedur_harmonic_magnitudes import VedurHarmonicMagnitudes


class VedurHarmonicMagnitudesGraph(Graph):

    # -------------------------------------------------------------------------
    #  Constants
    # -------------------------------------------------------------------------
    Y_LIM_TOP = 8000
    Y_LIM_BOT = -1000

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, filename: str, year: int):

        # init
        super().__init__(filename)
        self._year = year

        # clean image immediately
        self.__clean_image()

    def __clean_image(self):
        """Paint over certain parts of the image to avoid confusion of the data extraction process."""

        # get some reference coordinates
        bbox = self._graph_bbox
        hor_grid_lines = self._hor_grid_lines

        # prep drawing
        d = ImageDraw.Draw(self._pil_image)

        # draw white rectangle over plot legend at the top left
        rect = [(bbox.left + 1, bbox.top + 1), (bbox.right - 1, hor_grid_lines[0] - 1)]
        d.rectangle(rect, fill=VedurColors.WHITE.value)

    # -------------------------------------------------------------------------
    #  Public properties
    # -------------------------------------------------------------------------
    @cached_property
    def hor_scale(self) -> GraphScale:

        # extract dates from graph
        x_pos, dates = self._date_references

        # error detection
        if len(x_pos) < 2:
            raise RuntimeError(f"Could not reliably detect dates on x-scale of graph.")

        if (dates[-1] - dates[0]) < datetime.timedelta(days=7):
            raise RuntimeError(f"Reliably detected dates insufficiently spaced: {dates[0]} vs {dates[-1]}.")

        # keep first and last
        x_pos = [x_pos[0], x_pos[-1]]
        dates = [dates[0], dates[-1]]
        values = [ts_to_float(date) for date in dates]  # seconds since 2021.1.1

        return GraphScale(x_pos, values)

    @cached_property
    def vert_scale(self) -> GraphScale:
        return GraphScale([self._graph_bbox.top, self._graph_bbox.bottom], [self.Y_LIM_TOP, self.Y_LIM_BOT])

    @cached_property
    def data(self, ts: datetime.timedelta = datetime.timedelta(seconds=900)) -> VedurHarmonicMagnitudes:

        # signal bands
        blue, green, purple = self.__extract_signal_bands()

        # post-process
        self.__postprocess_signal_bands(blue, green, purple)

        # construct VedurHarmonicMagnitudes object, resample & return
        return VedurHarmonicMagnitudes(low=purple, mid=green, hi=blue).resample(ts)

    # -------------------------------------------------------------------------
    #  Data extraction
    # -------------------------------------------------------------------------
    @staticmethod
    def __postprocess_signal_bands(blue: MinMidMaxTimeSeries, green: MinMidMaxTimeSeries, purple: MinMidMaxTimeSeries):
        """Post-process min, mid, max of all 3 colors to fill in missing values."""

        # --- BLUE ------------------------------
        for i in range(1, blue.mid.data.size):
            if np.isnan(blue.mid.data[i]) and not np.isnan(blue.mid.data[i - 1]):
                # strategy: repeat previous values
                # (note: blue.min and blue.max are always both NaN or both non-NaN)
                blue.min.data[i] = blue.min.data[i - 1]
                blue.max.data[i] = blue.max.data[i - 1]
                blue.mid.data[i] = blue.mid.data[i - 1]  # repeat previous value if missing

        # --- GREEN -----------------------------
        for i in range(1, green.mid.data.size):
            if np.isnan(green.mid.data[i]) and not np.isnan(green.mid.data[i - 1]):
                # strategy: repeat previous value and make sure it lies inside blue range (because it's hidden by it)
                # (note: green.min and green.max are always both NaN or both non-NaN)
                green.min.data[i] = np.clip(green.min.data[i - 1], blue.min.data[i], blue.max.data[i])
                green.max.data[i] = np.clip(green.max.data[i - 1], blue.min.data[i], blue.max.data[i])
                green.mid.data[i] = np.clip(green.mid.data[i - 1], blue.min.data[i], blue.max.data[i])

        # --- PURPLE -----------------------------
        blue_green_range = MinMidMaxTimeSeries(
            min=TimeSeries(blue.t0, blue.ts, np.minimum(blue.min.data, green.min.data)),
            mid=TimeSeries(blue.t0, blue.ts, (blue.mid.data + green.mid.data) / 2),
            max=TimeSeries(blue.t0, blue.ts, np.maximum(blue.max.data, green.max.data)),
        )
        for i in range(1, purple.mid.data.size):
            if np.isnan(purple.mid.data[i]) and not np.isnan(purple.mid.data[i - 1]):
                # strategy: repeat previous value and make sure it lies inside blue & green range (because it's hidden by them)
                # (note: purple.min and purple.max are always both NaN or both non-NaN)
                purple.min.data[i] = np.clip(
                    purple.min.data[i - 1], blue_green_range.min.data[i], blue_green_range.max.data[i]
                )
                purple.max.data[i] = np.clip(
                    purple.max.data[i - 1], blue_green_range.min.data[i], blue_green_range.max.data[i]
                )
                purple.mid.data[i] = np.clip(
                    purple.mid.data[i - 1], blue_green_range.min.data[i], blue_green_range.max.data[i]
                )

    # -------------------------------------------------------------------------
    #  Extract raw data signals
    # -------------------------------------------------------------------------
    def __extract_signal_bands(self) -> Tuple[MinMidMaxTimeSeries, MinMidMaxTimeSeries, MinMidMaxTimeSeries]:
        """Returns a MinMidMaxTimeSeries object for each of the 3 signals in the graph (blue, green, purple)."""

        # extract x-range
        x_min, x_max = self.__extract_signals_x_range()

        # extract raw ranges
        blue = self.__extract_single_signal_band(self.__blue_pixels, x_min, x_max)
        green = self.__extract_single_signal_band(self.__green_pixels, x_min, x_max)
        purple = self.__extract_single_signal_band(self.__purple_pixels, x_min, x_max)

        # return result
        return blue, green, purple

    def __extract_signals_x_range(self) -> Tuple[int, int]:
        """
        Return left- and right-most pixel positions where one of the 3 signal colors is detected. Range is inclusive of end-points.
        """

        # True where we find any of the 3 colors
        blue_green_purple = self.__blue_pixels + self.__green_pixels + self.__purple_pixels  # type: np.ndarray

        # check which columns have at least 1 pixel of any of the 3 colors
        col_has_signal = blue_green_purple.max(axis=0, initial=False)
        cols_with_signal = [i for i, j in enumerate(col_has_signal) if j]

        # return left- and right-most pixel positions
        return min(cols_with_signal), max(cols_with_signal)

    def __extract_single_signal_band(self, pixels: np.ndarray, x_min: int, x_max: int) -> MinMidMaxTimeSeries:

        # --- time scale ----------------------------------
        t0 = float_to_ts(self.hor_scale.pixel_to_value(x_min))
        ts = datetime.timedelta(seconds=self.hor_scale.c1)

        # --- detect min & max ----------------------------
        n_cols = (x_max - x_min) + 1
        y_max = np.full(n_cols, fill_value=np.nan)
        y_min = np.full(n_cols, fill_value=np.nan)
        y_scale = self.vert_scale

        for i in range(n_cols):
            x_pos = x_min + i

            pixel_col = pixels[:, x_pos].flatten()
            rows_with_color = [i for i, j in enumerate(pixel_col) if j]

            if len(rows_with_color) > 0:
                y_max[i] = y_scale.pixel_to_value(min(rows_with_color))
                y_min[i] = y_scale.pixel_to_value(max(rows_with_color))

        return MinMidMaxTimeSeries(
            min=TimeSeries(t0, ts, y_min), mid=TimeSeries(t0, ts, (y_min + y_max) / 2), max=TimeSeries(t0, ts, y_max)
        )

    # -------------------------------------------------------------------------
    #  Detect data signal pixels
    # -------------------------------------------------------------------------
    @cached_property
    def __blue_pixels(self) -> np.ndarray:
        return self._detect_color(clr=VedurColors.BLUE.value, clr_tol=0.05)

    @cached_property
    def __green_pixels(self) -> np.ndarray:
        return self._detect_color(clr=VedurColors.GREEN.value, clr_tol=0.05)

    @cached_property
    def __purple_pixels(self) -> np.ndarray:
        return self._detect_color(clr=VedurColors.PURPLE.value, clr_tol=0.05)

    # -------------------------------------------------------------------------
    #  Detect grid & bbox
    # -------------------------------------------------------------------------
    @cached_property
    def _graph_bbox(self) -> BBox:
        """returns (left, right, top, bottom) bounding box of the actual graph inside the image."""
        left_right = sorted(self._detect_vert_lines(clr=VedurColors.BLACK.value, min_frac=0.8))
        top_bottom = sorted(self._detect_hor_lines(clr=VedurColors.BLACK.value, min_frac=0.8))
        return BBox(left=left_right[0], right=left_right[1], top=top_bottom[0], bottom=top_bottom[1])

    @cached_property
    def _hor_grid_lines(self) -> List[int]:
        """
        List of (sorted) pixel positions of horizontal grid lines (grey dashed lines).
        We could miss 1 or 2 grid lines in case the graph lines cover too much of some lines.
        """
        return sorted(self._detect_hor_lines(clr=VedurColors.GREY.value, min_frac=0.1))

    @cached_property
    def _vert_grid_lines(self) -> List[int]:
        """
        List of (sorted) pixel positions of vertical grid lines (grey dashed lines).
        We could miss 1 or 2 grid lines in case the graph lines cover too much of some lines.
        """
        return sorted(self._detect_vert_lines(clr=VedurColors.GREY.value, min_frac=0.1))

    # -------------------------------------------------------------------------
    #  Date/Time handling
    # -------------------------------------------------------------------------
    @cached_property
    def _date_references(self) -> Tuple[List[int], List[datetime.date]]:
        """Returns 2 lists: x_pos & dates of equal lengths with corresponding hor. pixel positions and reliably detected dates."""

        # perform OCR and filter the results
        ocr_results = self._reliable_ocr_dates

        # match to vertical grid line positions
        vert_grid_lines = self._vert_grid_lines
        result_x_pos = []
        result_dates = []
        for x_pos in vert_grid_lines:

            matching_dates = [
                date for bbox, date, confidence in ocr_results if (bbox.left < x_pos) and (bbox.right > x_pos)
            ]

            if len(matching_dates) == 1:
                result_x_pos.append(x_pos)
                result_dates.append(matching_dates[0])

        return result_x_pos, result_dates

    @cached_property
    def _reliable_ocr_dates(self) -> List[Tuple[BBox, datetime.date, float]]:

        return [
            (bbox, self.__str_to_date(text), confidence)
            for bbox, text, confidence in self._ocr()
            if (bbox.top > self._graph_bbox.bottom) and (confidence >= 0.70) and self.__is_valid_date_str(text)
        ]

    def __str_to_date(self, s: str) -> Optional[datetime.date]:
        """Converts eg '26/08' to a datetime.date or None if the format does not fit."""

        # invalid length, format or characters
        if (len(s) != 5) or ("/" not in s) or any([char not in "0123456789/" for char in s]):
            return None

        # split and see if the parts look ok
        s_parts = s.split("/")
        if (len(s_parts) != 2) or any([len(s) != 2 for s in s_parts]):
            return None

        # convert to date, but detect any errors (day or month might be outside acceptable range)
        day = int(s_parts[0])
        month = int(s_parts[1])
        try:
            return datetime.date(self._year, month, day)
        except:
            return None

    def __is_valid_date_str(self, s: str) -> bool:
        return self.__str_to_date(s) is not None

    # -------------------------------------------------------------------------
    #  Debug
    # -------------------------------------------------------------------------
    def debug_image(
        self,
        show_grids: bool = False,
        show_dates: bool = False,
        show_blue: bool = False,
        show_green: bool = False,
        show_purple: bool = False,
    ) -> Image.Image:
        """
        Generates an image with annotations superimposed on the original image,
          to debug the data extraction process and verify everything went well.
        :param show_grids: (bool) show extracted x- and y-grids & scales
        :param show_dates: (bool) show detected dates
        :param show_blue: (bool) show blue data points.
        :param show_green: (bool) show green data points.
        :param show_purple: (bool) show purple data points.
        :return: Pillow image object.
        """

        # --- original image ------------------------------
        img = self._pil_image.copy()  # type: Image.Image
        img = ImageEnhance.Contrast(img).enhance(0.25)

        drw = ImageDrawPlus(img)

        # --- plot gridlines ------------------------------
        if show_grids:
            bbox = self._graph_bbox
            x_grid_pos = self._vert_grid_lines
            y_grid_pos = self._hor_grid_lines

            drw.rectangle_from_bbox(bbox, fill=None, outline=VedurColors.RED.value)

            for x in x_grid_pos:
                drw.line_with_style(
                    xy_from=(x, bbox.top), xy_to=(x, bbox.bottom), fill=VedurColors.RED.value, style="--"
                )

            for y in y_grid_pos:
                drw.line_with_style(
                    xy_from=(bbox.left, y), xy_to=(bbox.right, y), fill=VedurColors.RED.value, style="--"
                )

        # --- highlight detected dates --------------------
        if show_dates:
            for bbox, date, confidence in self._reliable_ocr_dates:
                drw.rectangle_from_bbox(bbox, outline=VedurColors.RED.value)
                drw.default_text(
                    xy=(bbox.left, bbox.top - 9), text=date.strftime("%d.%m.%Y"), fill=VedurColors.RED.value, size=7
                )

        # --- show data -----------------------------------
        if show_blue or show_green or show_purple:

            # extract data & range
            x_range = self.__extract_signals_x_range()
            y_scale = self.vert_scale
            blue, green, purple = self.__extract_signal_bands()
            self.__postprocess_signal_bands(blue, green, purple)

            # show purple
            if show_purple:
                for i in range(len(purple.time)):
                    x_pos = x_range[0] + i
                    if not np.isnan(purple.min.data[i]):
                        drw.point(xy=(x_pos, y_scale.value_to_pixel(purple.min.data[i])), fill=VedurColors.PURPLE.value)
                    if not np.isnan(purple.mid.data[i]):
                        drw.point(xy=(x_pos, y_scale.value_to_pixel(purple.mid.data[i])), fill=VedurColors.PURPLE.value)
                    if not np.isnan(purple.max.data[i]):
                        drw.point(xy=(x_pos, y_scale.value_to_pixel(purple.max.data[i])), fill=VedurColors.PURPLE.value)

            # show green
            if show_green:
                for i in range(len(green.time)):
                    x_pos = x_range[0] + i
                    if not np.isnan(green.min.data[i]):
                        drw.point(xy=(x_pos, y_scale.value_to_pixel(green.min.data[i])), fill=VedurColors.GREEN.value)
                    if not np.isnan(green.mid.data[i]):
                        drw.point(xy=(x_pos, y_scale.value_to_pixel(green.mid.data[i])), fill=VedurColors.GREEN.value)
                    if not np.isnan(green.max.data[i]):
                        drw.point(xy=(x_pos, y_scale.value_to_pixel(green.max.data[i])), fill=VedurColors.GREEN.value)

            # show blue
            if show_blue:
                for i in range(len(blue.time)):
                    x_pos = x_range[0] + i
                    if not np.isnan(blue.min.data[i]):
                        drw.point(xy=(x_pos, y_scale.value_to_pixel(blue.min.data[i])), fill=VedurColors.BLUE.value)
                    if not np.isnan(blue.mid.data[i]):
                        drw.point(xy=(x_pos, y_scale.value_to_pixel(blue.mid.data[i])), fill=VedurColors.BLUE.value)
                    if not np.isnan(blue.max.data[i]):
                        drw.point(xy=(x_pos, y_scale.value_to_pixel(blue.max.data[i])), fill=VedurColors.BLUE.value)

        return img
