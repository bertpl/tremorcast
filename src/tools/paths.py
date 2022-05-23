import os
from pathlib import Path


def get_git_root() -> str:
    """To be used to get the GIT root from the current working directory, e.g. from inside a notebook"""

    FILE_ONLY_IN_GIT_ROOT = "pyproject.toml"

    p = Path(os.getcwd())
    while not (p / FILE_ONLY_IN_GIT_ROOT).is_file():
        if p.parent != p:
            p = p.parent
        else:
            raise ValueError(f"could not determine git root from path {os.getcwd()}")

    return str(p)
