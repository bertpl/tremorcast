# TremorCast
Python project for volcanic tremor forecasting, inspired by Fagradalsfjall, Iceland and the publically available vedur.is tremor plots.


## 1. Branching structure

For now a very simple structure is used:
* `main`: main branch where the most recent working version can be found; target of feature branch merges.
* `feature/*`: work on additional features / changes, directly merged to `main`.

If the need arises to go for official release cycles, this branching structure can be revised.

## 2. Project structure

The top-level structure can be described as follows:
* `debug`: for local debugging / testing needs
* `src`: contains the main code base, with sub-folders (most generic first):
    * `tools`: smaller, independent tools to be used in other parts of the code base
    * `base`: core functionality implemented in a reusable fashion to support specific functionality in other folders
    * `applications`: functionality implemented here is targeted at a specific application area, e.g. `vedur_is`, which is aimed at Icelandic volcano forecasting based on vedur.is data.
    * `projects`: these are specific projects, building on top of functionality in the previous 2 folders.

## 3. Getting started

* git clone
* pip install -r requirements.txt
* `tremorcast.py` (currently empty) is the main future entrypoint
* local debugging can be done in the `debug` folder, where you can copy `debug_template.py` to `debug.py` (.gitignored) for playing around with the code base locally.




