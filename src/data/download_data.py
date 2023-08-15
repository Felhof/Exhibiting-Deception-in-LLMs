# -*- coding: utf-8 -*-
import click
import requests
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


TQA_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
TQA_AUGMENTED_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/data/finetune_truth.jsonl"

TQA_FILENAME = "TruthfulQA.csv"
TQA_AUGMENTED_FILENAME = "TQA_augmented.json"


@click.command()
@click.argument("output_dir", type=click.Path())
def main(output_dir=""):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"downloading datasets to `{output_dir}`")

    urls = [
        TQA_URL,
        TQA_AUGMENTED_URL
    ]
    filenames = [
        TQA_FILENAME,
        TQA_AUGMENTED_FILENAME
    ]

    for url, filename in zip(urls, filenames):
        output_filepath = f"{output_dir}/{filename}"

        response = requests.get(url)
        response.raise_for_status()

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(output_filepath, "wb") as file:
            file.write(response.content)

        logger.info(f"{filename} downloaded and saved successfully to `{output_filepath}`")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
