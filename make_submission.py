#!/bin/python3

# AUTHOR b.covas (bernardo.covas@outlook.com)

# This script is used to perform a submission.
# It saves it in csv format in accordance to the
# competition rules, to a filename controlled
# by arguments, in the submission directory
# in PATHS.json.

# NOTE: This sctipt can be used in combination with
# validate.py for easy experimenting.
# Ex: "python .\make_submission.py test && python .\validate.py submission.csv"
# performs an evaluation step.

import os
import time
import logging
import multiprocessing

import tensorflow as tf
import numpy as np

from package import common, dataset as dataset_module, model as model_module


def make_submission(
        tfrecord_name: str,
        submission_fname: str,
        batch_size: int,
        paralell_calls: int):

    config = common.TFHubModels(common.ConfigurationJson().TF_HUB_MODULE)

    def _inp_fn():

        def map_fn(example):

            features = dataset_module.tf_parse_single_example(example, [
                dataset_module.TFRecordKeys.ID,
                dataset_module.TFRecordKeys.IMG_PATHS
            ])

            img_id = features[dataset_module.TFRecordKeys.ID]
            paths = features[dataset_module.TFRecordKeys.IMG_PATHS]

            img = dataset_module.tf_load_image(paths)[:, :, 0:3]
            img = tf.to_float(img) / 255
            img = tf.image.random_crop(img, config.expected_image_size + (3,))

            return {
                dataset_module.TFRecordKeys.ID: img_id,
                dataset_module.TFRecordKeys.DECODED: img
            }

        dataset = tf.data.TFRecordDataset(tfrecord_name) \
            .map(map_fn, paralell_calls) \
            .batch(batch_size) \
            .prefetch(2)

        return dataset

    logger = logging.getLogger("Submittor")
    submission = common.Submission(submission_fname)

    estimator = model_module.ProteinEstimator()
    wanted_predictions = [
        model_module.ProteinEstimator.IMAGE_ID,
        model_module.ProteinEstimator.SCORES
    ]

    tf.logging.set_verbosity(tf.logging.INFO)
    for i, predictions in enumerate(
        estimator.predict(_inp_fn, wanted_predictions)):
        i += 1

        scores = predictions[model_module.ProteinEstimator.SCORES]
        img_id = predictions[model_module.ProteinEstimator.IMAGE_ID]

        img_id = img_id.decode()
        scores = (scores > 0.5).astype(np.int)
        label = common.one_hot_to_label(scores)

        if len(label) == 0:
            label = [str(np.argmax(scores))]

        submission.add_submission(img_id, label)
        if i % 100 == 0:
            logger.info(f"Wrote {i} examples.")

    submission.end_sumbission()
    logger.info(f"Finished, Wrote {i} examples.")

def validate_sumbission(sample_subfile: str, subfile: str):

    logger = logging.getLogger("Validator")
    sample_subfile_csv = open(sample_subfile)
    subfile_csv = open(subfile)

    for sample_row, row in zip(sample_subfile_csv, subfile_csv):

        sample_id = sample_row.split(",")[0]
        img_id = row.split(",")[0]

        if not sample_id == img_id:
            logger.error(f"Mismached: {sample_id}")
            subfile_csv.close()
            sample_subfile_csv.close()
            return

        logger.info(f"Mached: {sample_id}")

if __name__ == "__main__":

    import argparse
    import sys

    paths = common.PathsJson()
    records={
        "train": paths.TRAIN_RECORD,
        "test": paths.TEST_RECORD,
        "predict": paths.PREDICT_FEATURES_RECORD
    }

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--batch_size", type=int, default=10, help=f"""
    Batch size to use. Defaults to 10.
    """)
    argparser.add_argument(
        "--paralell_calls", type=int, default=1, help=f"""
    Paralell calls for image decoding/feature reader.
    Defauts to 1 (Sequential).
    """)
    argparser.add_argument(
        "--feature_record", help=f"""
    What feature_record to use. Defaults to predict. {list(records.keys())}
    """)
    argparser.add_argument(
        "--validate", action="store_true", help=f"""
    Validate the prediction order with the sample csv. (If prediction is 'predict')
    """)
    argparser.add_argument(
        "--submission_name", default="submission.csv", help=f"""
    What to call the submission file. Defaults to 'submission.csv'.
    """)

    if len(sys.argv) < 2:
        argparser.print_help()
        exit(1)

    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MakeSubmission")
    tflogger = logging.getLogger("tensorflow")
    tf.logging.set_verbosity(tf.logging.WARN)
    tflogger.propagate = False

    record = records.get(args.feature_record)
    if record is None:
        logger.error(f"Unknown record: {record}. Use -h for help.")
        exit(1)

    if not os.path.exists(paths.SUBMISSION_DIR):
        os.makedirs(paths.SUBMISSION_DIR)

    sub_file = os.path.join(paths.SUBMISSION_DIR, args.submission_name)

    init = time.time()

    make_submission(
        record,
        sub_file,
        args.batch_size,
        args.paralell_calls)

    if args.validate:
        if args.feature_record == "predict":
            validate_sumbission(paths.CSV_TEST, sub_file)
        else:
            logger.error(f"Can't validate '{args.feature_record}'. Only 'predict'.")

    end = time.time()

    logger.info(f"""

    Submitted to {sub_file} 
    in {(end - init)} seconds.
    """)