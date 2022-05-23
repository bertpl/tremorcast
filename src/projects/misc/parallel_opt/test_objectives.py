from time import sleep
from typing import Optional

import numpy as np


def quadratic_2d(x: float, y: float) -> float:
    sleep(1.0)
    return (x**2) + 2 * (y**2)


def quadratic_5d(a: float, b: float, c: float, d: float, e: float) -> float:
    sleep(1.0)
    return (a**2) + 2 * (b**2) + 3 * (c**2) + 4 * (d**2) + 5 * (e**2)


def quadratic_5d_unreliabe(a: float, b: float, c: float, d: float, e: float) -> float:
    sleep(1.0)
    if np.random.random_sample() < 0.05:
        raise RuntimeError
    else:
        return (a**2) + 2 * (b**2) + 3 * (c**2) + 4 * (d**2) + 5 * (e**2)


def quadratic_5d_always_fails(a: float, b: float, c: float, d: float, e: float) -> float:
    sleep(1.0)
    if np.random.random_sample() < 2.0:
        raise RuntimeError
    else:
        return (a**2) + 2 * (b**2) + 3 * (c**2) + 4 * (d**2) + 5 * (e**2)


def quadratic_5d_can_return_none(a: float, b: float, c: float, d: float, e: float) -> Optional[float]:
    if max([a, b, c, d, e]) > 5:
        return None
    else:
        sleep(1.0)
        return (a**2) + 2 * (b**2) + 3 * (c**2) + 4 * (d**2) + 5 * (e**2)
