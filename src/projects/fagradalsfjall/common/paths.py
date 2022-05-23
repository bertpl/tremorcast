import os
from pathlib import Path

from src.projects.fagradalsfjall.common.project_settings import (
    PATH_BLOG_POST_1,
    PATH_BLOG_POST_2,
    PATH_BLOG_POST_3,
    PATH_BLOG_POST_4,
    PATH_BLOG_POST_5,
    PATH_BLOG_POST_6,
)


def get_blog_post_subfolder(post_nr: int, sub_folder: str) -> Path:

    all_paths = {
        1: PATH_BLOG_POST_1,
        2: PATH_BLOG_POST_2,
        3: PATH_BLOG_POST_3,
        4: PATH_BLOG_POST_4,
        5: PATH_BLOG_POST_5,
        6: PATH_BLOG_POST_6,
    }

    base_folder = Path(all_paths[post_nr])
    os.makedirs(base_folder, exist_ok=True)

    folder = base_folder / sub_folder
    os.makedirs(folder, exist_ok=True)

    return folder
