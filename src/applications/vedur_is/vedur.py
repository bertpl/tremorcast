from enum import Enum


class VedurFreqBands(Enum):
    LOW = "0.5-1.0 Hz"
    MID = "1-2 Hz"
    HI = "2-4 Hz"

    def __str__(self):
        return self.name.lower()


class VedurColors(Enum):
    # chart colors - signals
    PURPLE = (148, 0, 211)  # 0.5-1.0Hz data
    GREEN = (0, 158, 115)  # 1-2Hz data
    BLUE = (86, 180, 233)  # 2-4Hz data
    # chart colors - other
    BLACK = (0, 0, 0)  # text & chart outline
    WHITE = (255, 255, 255)  # background
    GREY = (160, 160, 160)  # chart gridlines
    # debugging
    RED = (255, 0, 0)  # for debugging purposes


VEDUR_DATA_COLORS = {
    VedurFreqBands.LOW: VedurColors.PURPLE,
    VedurFreqBands.MID: VedurColors.GREEN,
    VedurFreqBands.HI: VedurColors.BLUE,
}
