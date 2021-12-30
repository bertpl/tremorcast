import os
import pickle
from typing import Optional

from src.applications.vedur_is import VedurHarmonicMagnitudes, VedurHarmonicMagnitudesGraph

from ._project_settings import FILE_DATASET, FILE_DATASET_GRAPH, FILES_GRAPH, PATH_SOURCE_DEBUG


# -------------------------------------------------------------------------
#  PROCESSING
# -------------------------------------------------------------------------
def extract_data():

    # --- process all files -------------------------------
    all_data = None  # type: Optional[VedurHarmonicMagnitudes]
    for full_path in FILES_GRAPH:

        _, file_name = os.path.split(full_path)

        print(f"Extracting data from file {file_name} ...".ljust(60), end="")

        file_without_ext, _ = os.path.splitext(file_name)
        graph = VedurHarmonicMagnitudesGraph(full_path, year=2021)

        graph.debug_image(show_grids=True).save(os.path.join(PATH_SOURCE_DEBUG, file_without_ext + "_grids.png"))
        graph.debug_image(show_dates=True).save(os.path.join(PATH_SOURCE_DEBUG, file_without_ext + "_dates.png"))
        graph.debug_image(show_blue=True).save(os.path.join(PATH_SOURCE_DEBUG, file_without_ext + "_data_blue.png"))
        graph.debug_image(show_green=True).save(os.path.join(PATH_SOURCE_DEBUG, file_without_ext + "_data_green.png"))
        graph.debug_image(show_purple=True).save(os.path.join(PATH_SOURCE_DEBUG, file_without_ext + "_data_purple.png"))

        if all_data is None:
            all_data = graph.data
        else:
            all_data |= graph.data

        print("Done.")

    print()

    # --- save figure & data ------------------------------
    print(f"Saving to disk ...".ljust(60), end="")

    # figure
    fig, ax = all_data.create_plot(title="Fagradalsfjall (faf)")
    fig.savefig(FILE_DATASET_GRAPH, dpi=450)

    # data
    save_data(all_data)

    print("Done.")


# -------------------------------------------------------------------------
#  LOAD / SAVE
# -------------------------------------------------------------------------
def save_data(data: VedurHarmonicMagnitudes):

    with open(FILE_DATASET, "wb") as f:
        pickle.dump(data, f)


def load_data() -> VedurHarmonicMagnitudes:

    with open(FILE_DATASET, "rb") as f:
        data = pickle.load(f)

    return data
