import datetime
import os

from src.applications.vedur_is.vedur import VEDUR_DATA_COLORS, VedurFreqBands
from src.base.forecasting.evaluation.cross_validation import TimeSeriesCVSplitter
from src.base.forecasting.evaluation.metrics.tabular_metrics import RMSE
from src.base.forecasting.evaluation.metrics.timeseries_metrics import AreaUnderCurveLogLog, MaxAccurateLeadTime
from src.tools.paths import get_git_root

# =================================================================================================
#  Core folder structure
# =================================================================================================
PATH_PROJECT_ROOT = os.path.join(get_git_root(), "_data/projects/fagradalsfjall")

PATH_SCRAPED = os.path.join(PATH_PROJECT_ROOT, "0_scraped")
PATH_SCRAPED_DEBUG = os.path.join(PATH_SCRAPED, "debug")
PATH_DATASET = os.path.join(PATH_PROJECT_ROOT, "1_dataset")
PATH_BLOG_POSTS = os.path.join(PATH_PROJECT_ROOT, "2_blog_posts")
PATH_MODEL_REPO = os.path.join(PATH_PROJECT_ROOT, "3_model_repo")

PATH_BLOG_POST_1 = os.path.join(PATH_BLOG_POSTS, "post_1_intro")
PATH_BLOG_POST_2 = os.path.join(PATH_BLOG_POSTS, "post_2_data_extraction")
PATH_BLOG_POST_3 = os.path.join(PATH_BLOG_POSTS, "post_3_naive_models")
PATH_BLOG_POST_4 = os.path.join(PATH_BLOG_POSTS, "post_4_arma")
PATH_BLOG_POST_5 = os.path.join(PATH_BLOG_POSTS, "post_5_n_step")
PATH_BLOG_POST_6 = os.path.join(PATH_BLOG_POSTS, "post_6_nn")


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
FILE_DATASET_CROSS_VALIDATION = os.path.join(PATH_DATASET, "dataset_cv")

# --- settings --------------------------------------------
# TOTAL NUMBER OF DAYS = 45
DATASET_TRAIN_TEST_TS_FROM = datetime.datetime(2021, 7, 20, 0, 0)  # start of long, clean stretch of data
DATASET_TRAIN_TEST_TS_TO = datetime.datetime(2021, 9, 3, 0, 0)  # end of long, clean stretch of data

# train & test sample ranges
# TRAIN: first 31 days (1 month)
DATASET_TRAIN_N_DAYS = 31  # ±70%
DATASET_TEST_N_DAYS = 14  # ±30%
DATASET_TRAIN_N_SAMPLES = DATASET_TRAIN_N_DAYS * 96
DATASET_TEST_N_SAMPLES = DATASET_TEST_N_DAYS * 96

DATASET_TRAIN_SAMPLE_FROM = 0
DATASET_TRAIN_SAMPLE_TO = DATASET_TRAIN_N_SAMPLES

# TEST: last 14 days
DATASET_TEST_SAMPLE_FROM = DATASET_TRAIN_SAMPLE_TO
DATASET_TEST_SAMPLE_TO = DATASET_TEST_SAMPLE_FROM + DATASET_TEST_N_SAMPLES

# -------------------------------------------------------------------------
#  FORECASTING PROBLEM DEFINITION / CROSS VALIDATION
# -------------------------------------------------------------------------

# --- CV settings -----------------------------------------
CV_MIN_SAMPLES_TRAIN = 20 * 96  # 20 days
CV_MIN_SAMPLES_VALIDATE = 3 * 96  # 3 days
CV_N_SPLITS = 10
TS_CV_SPLITTER = TimeSeriesCVSplitter(CV_MIN_SAMPLES_TRAIN, CV_MIN_SAMPLES_VALIDATE, CV_N_SPLITS)

# size of RMSE curve to use for computing cv metrics
CV_HORIZON_SAMPLES = 2 * 96  # 2 days

# stride for validation simulations (=visual inspection)
SIMULATION_STRIDE = 48  # 12h

# --- signals ---------------------------------------------

# -- middle of purple line --
# FORECAST_SIGNAL_NAME = "low_mid"
# FORECAST_SIGNAL_COLOR = VEDUR_DATA_COLORS[VedurFreqBands.LOW]
# CV_METRIC_RMSE_THRESHOLD = 300

# -- bottom of blue line --
FORECAST_SIGNAL_NAME = "hi_min"
FORECAST_SIGNAL_COLOR = VEDUR_DATA_COLORS[VedurFreqBands.HI].value
CV_METRIC_RMSE_THRESHOLD = 1000

# --- define metrics --------------------------------------
TABULAR_METRIC = RMSE()

TS_ALL_METRICS = [AreaUnderCurveLogLog(TABULAR_METRIC), MaxAccurateLeadTime(TABULAR_METRIC, CV_METRIC_RMSE_THRESHOLD)]

TS_PRIMARY_METRIC = TS_ALL_METRICS[0]
TS_PRIMARY_METRIC_DISPLAY_NAME = "Area Under Curve\n(log-log RMSE-vs-lead-time curve)"
