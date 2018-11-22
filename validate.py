import logging
import os

import tensorflow as tf

from package import \
    common, \
    model as model_module, \
    dataset as dataset_module

def main(dataset: dataset_module.Dataset, sumbission_file: str):

    logger = logging.getLogger("Validate")
    n_correct = 0
    n = 0
    with open(sumbission_file) as f:
        f.__next__()
        for line in f:
            img_id, label = line.split(",")
            label = label.replace("\n", "").split(" ")
            label = sorted(list(map(str, label)))
            true_label = sorted(dataset.label(img_id))

            n += 1
            if label == true_label:
                n_correct += 1

            logger.info(f"{label} -> {true_label}")

    return n, n_correct

if __name__ == "__main__":

    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_file", help="""
    The name of the prediction file.
    """)

    logger = logging.getLogger("Validator")
    pathsJson = common.PathsJson()
    dataset = dataset_module.Dataset(None, pathsJson.CSV_TRAIN)
    args = parser.parse_args()

    prediction_file = os.path.join(
        pathsJson.SUBMISSION_DIR, args.prediction_file)

    if not os.path.exists(prediction_file):
        logger.error(f"""
    {args.prediction_file} does not exist in {pathsJson.SUBMISSION_DIR}
    """)
        exit(1)

    n, nc = main(dataset, prediction_file)
    logger.info(f"""

    From {n} examples, {nc} are correct.
    ({round((nc / n) * 100, 3)}%)
    """)