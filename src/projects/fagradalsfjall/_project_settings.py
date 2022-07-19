import os

from src.tools.paths import get_git_root

# -------------------------------------------------------------------------
#  Folders
# -------------------------------------------------------------------------
PATH_PROJECT_ROOT = os.path.join(get_git_root(), "_data/projects/fagradalsfjall")

PATH_SCRAPED = os.path.join(PATH_PROJECT_ROOT, "0_scraped")
PATH_SCRAPED_DEBUG = os.path.join(PATH_SCRAPED, "debug")
PATH_DATASET = os.path.join(PATH_PROJECT_ROOT, "1_dataset")
PATH_BLOG_POSTS = os.path.join(PATH_PROJECT_ROOT, "2_blog_posts")


# -------------------------------------------------------------------------
#  SOURCE FILES
# -------------------------------------------------------------------------
SCRAPED_FILES_FOR_PROCESSING = [
    os.path.join(PATH_SCRAPED, filename)
    for filename in [
        "faf_20210722_2330_long.png",
        "faf_20210818_2330_long.png",
        "faf_20210825_0000.png",
        "faf_20210828_1200.png",
        "faf_20210829_1510.gif",
        "faf_20210901_0900.gif",
        "faf_20210905_1000.gif",
        "faf_20210912_1000.gif",
        "faf_20210918_1300.gif",
        "faf_20210926_1500.gif",
        "faf_20211001_2000.gif",
        "faf_20211010_1400.gif",
        "faf_20211016_1300.gif",
    ]
]

# -------------------------------------------------------------------------
#  DATASET
# -------------------------------------------------------------------------

# STEP 0 - extract data from plots
FILE_DATASET_FULL = os.path.join(PATH_DATASET, "dataset_full")

# STEP 1 - prepare train & test set
FILE_DATASET_SELECTION = os.path.join(PATH_DATASET, "dataset_selection")
FILE_DATASET_TRAIN = os.path.join(PATH_DATASET, "dataset_train")
FILE_DATASET_TEST = os.path.join(PATH_DATASET, "dataset_test")

FORECAST_SIGNAL_NAME = "low_mid"
FORECAST_MAE_THRESHOLD = 200
