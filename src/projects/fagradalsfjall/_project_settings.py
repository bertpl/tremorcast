import os
import datetime

from src.tools.paths import get_git_root

# =================================================================================================
#  Core folder structure
# =================================================================================================
PATH_PROJECT_ROOT = os.path.join(get_git_root(), "_data/projects/fagradalsfjall")

PATH_SCRAPED = os.path.join(PATH_PROJECT_ROOT, "0_scraped")
PATH_SCRAPED_DEBUG = os.path.join(PATH_SCRAPED, "debug")
PATH_DATASET = os.path.join(PATH_PROJECT_ROOT, "1_dataset")
PATH_BLOG_POSTS = os.path.join(PATH_PROJECT_ROOT, "2_blog_posts")


# =================================================================================================
#  BLOG 2 - Construct dataset
# =================================================================================================

# -------------------------------------------------------------------------
#  INPUT - scraped files
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
#  OUTPUT - full dataset
# -------------------------------------------------------------------------
FILE_DATASET_FULL = os.path.join(PATH_DATASET, "dataset_full")

# =================================================================================================
#  BLOG 3 - Train & test split + naive models
# =================================================================================================

# -------------------------------------------------------------------------
#  DATASET - train / test split
# -------------------------------------------------------------------------

# --- paths -----------------------------------------------
FILE_DATASET_SELECTION = os.path.join(PATH_DATASET, "dataset_selection")
FILE_DATASET_TRAIN = os.path.join(PATH_DATASET, "dataset_train")
FILE_DATASET_TEST = os.path.join(PATH_DATASET, "dataset_test")

# --- settings --------------------------------------------
# TOTAL NUMBER OF DAYS = 45
DATASET_TRAIN_TEST_TS_FROM = datetime.datetime(2021, 7, 20, 0, 0)   # start of long, clean stretch of data
DATASET_TRAIN_TEST_TS_TO = datetime.datetime(2021, 9, 3, 0, 0)   # end of long, clean stretch of data

# train & test sample ranges
# TRAIN: first 31 days (1 month)
DATASET_TRAIN_SAMPLE_FROM = 0
DATASET_TRAIN_SAMPLE_TO = 31 * 96

# TEST: last 14 days
DATASET_TEST_SAMPLE_FROM = 31 * 96
DATASET_TEST_SAMPLE_TO = 45 * 96

# -------------------------------------------------------------------------
#  FORECASTING PROBLEM DEFINITION
# -------------------------------------------------------------------------

# --- paths -----------------------------------------------
FORECAST_SIGNAL_NAME = "low_mid"
FORECAST_MAE_THRESHOLD = 200
