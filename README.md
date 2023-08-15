# G5 - Exhibiting and mitigating deception in language models

Uses python `3.8`.

## Pipeline

This repo uses [make](https://linuxhint.com/make-command-linux/) to manage the pipeline and pipenv for dependency and environment management.

Install make and run the `make` command to see all the available "rules".

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make requirements`, `make data` or `make train`.
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see [sphinx-doc.org](sphinx-doc.org) for details.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting.
    │
    ├── Pipfile            <- Package dependencies used by pipenv.
    ├── Pipfile.lock       <- Dependency versions and hashes (similar to requirements.txt).
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported.
    ├── test_environment.py<- For now checks your version of python.
    │
    ├── tests              <- tests for scripts.
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module.
    │   │
    │   ├── data           <- Scripts to download or generate data.
    │   │   └── process_data.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling.
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions.
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations.
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see [tox.readthedocs.io](tox.readthedocs.io).

## Installing dependencies

1. Create the environment using `make create_environment`.
2. Install dependencies into this environment using `make requirements`.
3. Activate the environment to access dependencies using `pipenv shell`.

`Pipfile` and `Pipfile.lock` contain the list of dependencies

## Running tests

This project uses pytest.

To execute all tests run `make tests`.

## Formatting

This project uses black and isort.

To reformat all code in the repo run `make format`.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
