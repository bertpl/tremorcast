from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class TimeSeriesCVSplitter:

    min_samples_train: int
    min_samples_validate: int
    n_splits: int = 10

    def get_splits(self, n_samples_tot: int) -> List[Tuple[int, int]]:
        # returns a list of length n_splits containing (n_samples_train, n_samples_val)-tuples
        # consistent with requirements and available data

        # total number of samples that we can use in the validation sets
        n_samples_validation_tot = n_samples_tot - self.min_samples_train

        if n_samples_validation_tot >= self.n_splits * self.min_samples_validate:
            # validation sets do not overlap and are >= min_samples_validate

            val_set_sizes = np.diff(np.round(np.linspace(0, n_samples_tot - self.min_samples_train, self.n_splits + 1)))
            return [
                (int(self.min_samples_train + sum(val_set_sizes[:i])), int(val_set_sizes[i]))
                for i in range(self.n_splits)
            ]

        else:
            # validation sets overlap and are == min_samples_validate

            n_train_samples = np.round(
                np.linspace(self.min_samples_train, n_samples_tot - self.min_samples_validate, self.n_splits)
            )

            if len(set(n_train_samples)) < self.n_splits:
                raise ValueError(f"Cannot generate {self.n_splits} unique splits for given settings; not enough data.")

            return [(int(n_train), self.min_samples_validate) for n_train in n_train_samples]
