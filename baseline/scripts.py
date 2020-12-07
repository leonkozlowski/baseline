"""Utility Modules."""
import argparse
import csv
import logging
import os
import pickle

from bson import ObjectId
from pymongo import MongoClient

from baseline.config import (
    MONGODB_COL_ANNOTATION,
    MONGODB_COL_CONTEXT,
    MONGODB_CONNECTION,
    MONGODB_WBN,
)

logging.basicConfig(level="INFO")
_LOGGER = logging.getLogger(__name__)


def create_wbn_dataset(size: int = 200):
    """Generates dataset of annotated documents."""
    client = MongoClient(MONGODB_CONNECTION)

    # Establish database
    database = client[MONGODB_WBN]

    # Load collections
    annotations = database[MONGODB_COL_ANNOTATION]
    contexts = database[MONGODB_COL_CONTEXT]

    # Fetches any documents within limit
    keywords = list()
    classifications = list()
    cursor = annotations.find({}).limit(size)
    for document in cursor:
        # Annotated classification
        classification = (
            document.get("data", {})
            .get("eventData", {})
            .get("classification")
        )
        keywords_array = (
            document.get("data", {}).get("eventData", {}).get("keywords")
        )
        context_id = (
            document.get("data", {}).get("eventData", {}).get("context")
        )
        context = contexts.find_one({"_id": ObjectId(context_id)})
        # Tokens of news story
        tokens = context.get("data", {}).get("eventData", {}).get("tokens")
        keywords.append((tokens, keywords_array))
        classifications.append(classification)

    output = {
        "data": keywords,
        "target": classifications,
    }

    with open("data/pr-newswire.pickle", "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return os.path.exists("data/pr-newswire.pickle")


def create_dataset(size: int, output_path: str):
    """Generates dataset of annotated documents."""
    client = MongoClient(MONGODB_CONNECTION)

    # Establish database
    database = client[MONGODB_WBN]

    # Load collections
    annotations = database[MONGODB_COL_ANNOTATION]
    contexts = database[MONGODB_COL_CONTEXT]

    # Fetches any documents within limit
    dataset = list()
    cursor = annotations.find({}).limit(size)
    for document in cursor:
        # Annotated classification
        classification = (
            document.get("data", {})
            .get("eventData", {})
            .get("classification")
        )
        context_id = (
            document.get("data", {}).get("eventData", {}).get("context")
        )
        context = contexts.find_one({"_id": ObjectId(context_id)})
        # Tokens of news story
        paragraphs = (
            context.get("data", {}).get("eventData", {}).get("paragraphs")
        )
        # Generate data and target
        dataset.append(
            {
                "data": " ".join(paragraphs),  # Join list of paragraphs
                "target": classification,
            }
        )

    with open(output_path, "w") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=dataset[0].keys())
        writer.writeheader()
        writer.writerows(dataset)

    return os.path.exists(output_path)


def process_cmd_args():
    """Argparse processor."""
    parser = argparse.ArgumentParser(description="Dataset Creator.")
    parser.add_argument(
        "-s",
        "--size",
        required=False,
        type=int,
        default=200,
        help="Size of dataset to be generated.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        required=True,
        help="Output path for dataset .csv to be generated.",
    )

    arguments = parser.parse_args()

    assert arguments.size <= 500, "Maximum size is 500"
    _LOGGER.info(f"Running dataset-creator with parameters: {arguments}")

    return arguments


def main():
    arguments = process_cmd_args()
    _LOGGER.info(f"Creating dataset of size: {arguments.size}")
    created = create_dataset(
        size=arguments.size, output_path=arguments.output_path
    )
    _LOGGER.info(f"Dataset created: {created}")


if __name__ == "__main__":
    main()

    create_wbn_dataset(size=300)
