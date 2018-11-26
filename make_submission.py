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

from package import common, dataset as protein_dataset, model as model_module

def run_prediction(
        pred_queue,
        tfrecord_name: str,
        batch_size: int,
        paralell_calls: int):


    with tf.Graph().as_default():
        sess = tf.Session()

        dataset = tf.data.TFRecordDataset(tfrecord_name) \
            .map(lambda x: protein_dataset.tf_parse_single_example(x, [
                protein_dataset.TFRecordKeys.ID_KEY,
                protein_dataset.TFRecordKeys.IMG_PATHS_KEY,
            ])) \
            .batch(batch_size) \
            .prefetch(2)

        img_id_tensor, paths_tensor = dataset.make_one_shot_iterator().get_next()

        model = model_module.ExportedModel()
        pred_tensor = model.load(sess, paths_tensor)

        while True:
            try:
                pred_queue.put(sess.run([img_id_tensor, pred_tensor]))
            except tf.errors.OutOfRangeError:
                pred_queue.put(None)

def make_submission(queue, submission_fname: str):

    logging.basicConfig(level=logging.INFO)

    submission = common.Submission(submission_fname)
    logger = logging.getLogger("Submittor")

    while True:

        preds = queue.get()

        if preds is None:
            break

        for img_id, scores in list(zip(*preds)):

            img_id = img_id.decode()
            scores = (scores > 0.5).astype(np.int)
            label = common.one_hot_to_label(scores)

            if len(label) == 0:
                label = [str(np.argmax(scores))]

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
        "--use_full_model", action="store_true", help=f"""
    Use the full model and consume a directory,
    or use only the classifier on cached image features.
    """)
    argparser.add_argument(
        "--batch", type=int, default=10, help=f"""
    Batch size to use. Defaults to 10.
    """)
    argparser.add_argument(
        "--paralell", type=int, default=1, help=f"""
    Paralell calls for image decoding/feature reader.
    Defauts to 1.
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
            queue, record, args.batch, args.paralell))

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
        if args.feature_record == "predict":
            validate_sumbission(paths.CSV_TEST, sub_file)
        else:
            logger.error(f"Can't validate '{args.feature_record}'. Only 'predict'.")

    end = time.time()

    logger.info(f"""

    Submitted to {sub_file} 
    in {(end - init)} seconds.
    """)