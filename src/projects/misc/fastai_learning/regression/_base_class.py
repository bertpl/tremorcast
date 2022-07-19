from abc import ABC, abstractmethod

import numpy as np


class Regressor(ABC):
    def __init__(self, name: str, n_features: int, n_targets: int):
        self.name = name
        self.n_features = n_features
        self.n_targets = n_targets

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass
