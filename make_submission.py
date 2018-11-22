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

from package.model import ClassifierModel
from package import common, dataset as protein_dataset

def run_prediction(
        pred_queue,
        tfrecord_name: str,
        batch_size: int,
        paralell_calls: int):

    # pylint: disable=E1129
    with tf.Graph().as_default():
        sess = tf.Session()

        model = ClassifierModel()
        model_config = common.TFHubModels(
            common.ConfigurationJson().TF_HUB_MODULE)

        dataset = tf.data.TFRecordDataset(tfrecord_name) \
            .map(protein_dataset.tf_parse_single_example) \
            .map(lambda x: (
                x[protein_dataset.TFRecordKeys.IMG_FEATURES],
                x[protein_dataset.TFRecordKeys.ID_KEY])) \
            .prefetch(batch_size) \
            .batch(batch_size)

        features_tensor, id_tensor = dataset.make_one_shot_iterator().get_next()
        features_tensor.set_shape([None, model_config.feature_vector_size])
        logits_tensor = model.predict(features_tensor)
        prediction_tensor = tf.sigmoid(logits_tensor)

        model.load(sess)

        while True:
            try:
                preds, img_ids = sess.run([prediction_tensor, id_tensor])
                pred_queue.put((preds, img_ids))

            except tf.errors.OutOfRangeError:
                return

def make_submission(queue, submission_fname: str):

    logging.basicConfig(level=logging.INFO)

    submission = common.Submission(submission_fname)
    logger = logging.getLogger("Submittor")

    while True:

        preds_img_ids = queue.get()

        if preds_img_ids is None:
            break

        preds, img_ids = preds_img_ids

        for pred, img_id in list(zip(preds, img_ids)):

            pred_bool = pred > 0.5
            label = common.one_hot_to_label(pred_bool)

            if len(label) == 0:
                label = [str(np.argmax(pred))]

            img_id = img_id.decode()

            submission.add_submission(img_id, label)
            logger.info(f"{img_id} -> {label}")

    submission.end_sumbission()
    logger.info("Done")

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
        "feature_record", help=f"""
    What feature_record to use. Defaults to predict. {list(records.keys())}
    """)
    argparser.add_argument("--validate", action="store_true", help=f"""
    Validate the prediction order with the sample csv.
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

    record = records.get(args.feature_record)
    if record is None:
        logger.error(f"Unknown record: {record}. Use -h for help.")
        exit(1)

    if not os.path.exists(paths.SUBMISSION_DIR):
        os.makedirs(paths.SUBMISSION_DIR)

    sub_file = os.path.join(paths.SUBMISSION_DIR, args.submission_name)
    queue = multiprocessing.Queue(10)

    producer = multiprocessing.Process(
        target=run_prediction, args=(
            queue, record, 100, 16))

    consumer = multiprocessing.Process(
        target=make_submission,
        args=(queue, sub_file))

    init = time.time()

    producer.start()
    consumer.start()

    producer.join()
    queue.put(None)
    consumer.join()

    if args.validate:
        validate_sumbission(paths.CSV_TRAIN, sub_file)

    end = time.time()

    logger.info(f"""

    Submitted to {sub_file} 
    in {(end - init)} seconds.
    """)