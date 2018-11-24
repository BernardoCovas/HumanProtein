#!/bin/python3

# AUTHOR b.covas (bernardo.covas@outlook.com)

# This script is meant to parse the data.
# It uses the package.dataset class. The script joins the
# different channels of the images, and runs them through
# the selected deep learning model to calculate and cache the deep
# image features.
# This script can be resource intensive.

# NOTE: By default, this script uses all the available cpus, and
# no GPU. if you have a GPU available, consider adding the --gpu argument.
# (Depends on tensorflow-gpu)

import os
import logging
import itertools
import queue
import threading
import multiprocessing

import numpy as np
import tensorflow_hub as tf_hub
import tensorflow as tf

from package import model as model_module
from package import common
from package import dataset as protein_dataset

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parse_data")


def single_consumer(
        dataset: protein_dataset.Dataset,
        example_queue: queue.Queue,
        extract_features: bool,
        paralell_calls: int,
        use_gpu: bool,
        batch_size: int):

    # NOTE (bcovas) for some reason passing
    # this class to a process destroys it.
    dataset.reload()
    tf_hub_module = common.TFHubModels(common.ConfigurationJson().TF_HUB_MODULE)

    # pylint: disable=E1129
    tf_graph = tf.Graph()
    with tf_graph.as_default():

        def _map_fn(img_id):

            img = protein_dataset.tf_imgid_to_img(
                img_id, dataset.directory)[:, :, 0:3]
            img = tf.image.random_crop(img, list(tf_hub_module.expected_image_size) + [3])
            img_bytes = tf.image.encode_png(tf.cast(img, tf.uint8))

            return (img / 255, img_bytes)

        ids_dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(dataset.img_ids, tf.string))
        img_dataset = ids_dataset.map(
            _map_fn, paralell_calls)

        img_id_dataset = tf.data.Dataset.zip((
            img_dataset, ids_dataset)) \
            .batch(batch_size) \
            .prefetch(2)

        (img_tensors, img_bytes_tensor), id_tensors \
            = img_id_dataset.make_one_shot_iterator().get_next()

        features_tensors = None
        if extract_features:
            features_tensors = tf_hub.Module(tf_hub_module.url)(img_tensors)

        sess = tf.Session(config=tf.ConfigProto(
            device_count={'GPU': 0 if not use_gpu else 1}
        ))

        sess.run(tf.global_variables_initializer())
        calls = [id_tensors, img_bytes_tensor]

        if extract_features:
            calls.append(features_tensors)

        while True:
            try:
                example_queue.put(sess.run(calls))
            except tf.errors.OutOfRangeError:
                break


def write(
        example_queue: queue.Queue,
        clean_data_dir: str,
        tfrecord_fname: str,
        label_dataset: protein_dataset.Dataset):

    logger = logging.getLogger("ExampleWriter")
    label_dataset.reload()
    num_examples = len(label_dataset.img_ids)

    with tf.python_io.TFRecordWriter(tfrecord_fname) as writer:

        i = 0
        while True:

            example = example_queue.get()
            if example is None:
                break

            for elements in zip(*example):

                one_hot = np.zeros(
                    [protein_dataset.common.NUM_CLASSES], np.int)
                img_labels = label_dataset.label(elements[0].decode())
                img_labels = list(map(int, img_labels))
                for label in img_labels:
                    one_hot[label] = 1

                serialized_example = protein_dataset.tf_write_single_example(
                    elements[0], one_hot, elements[2] if len(elements) == 3 else None)
                writer.write(serialized_example)

                with open(os.path.join(clean_data_dir,
                        elements[0].decode() + ".png"), "wb") as f:
                    f.write(elements[1])

                if i % 100 == 0:
                    logger.info(
                        f"Wrote {i} examples of {num_examples}")
                i += 1
    logger.info(f"Done. Wrote {i} examples.")


def splitrecords(record: str, train_record: str, test_record: str, train_proba: float):

    import random
    logger = logging.getLogger("Splitter")

    train_writer = tf.python_io.TFRecordWriter(train_record)
    test_writer = tf.python_io.TFRecordWriter(test_record)
    n_test = 0
    n_train = 0

    logger.info(f"Spliting at {train_proba}...")

    for record_str in tf.python_io.tf_record_iterator(record):

        if random.random() > train_proba:
            test_writer.write(record_str)
            n_test += 1
        else:
            train_writer.write(record_str)
            n_train += 1

    logger.info(f"""
    
    {str(n_train)} examples in:
        {train_record} 
    and {str(n_test)} examples in:
        {test_record}.
    """)


def main(
        dataset: protein_dataset.Dataset,
        clean_data_dir: str,
        tfrecord_path: str,
        paralell_calls: int,
        extract_features: bool,
        gpu: bool, batch_size: int):
    """
    Even though, in this case, writing the features to disk 
    barely takes any time at all, I usually separate the 
    writing process from the prediction process.
    In some setups this might massively improve performance.
    """

    logger = logging.getLogger("MainImageParser")
    logger.info(f"Using batch_size of {batch_size}")
    logger.info(f"Using {paralell_calls} paralell calls.")

    if extract_features:
        if gpu:
            logger.info("Using GPU for feature extraction.")
        else:
            logger.warn(
                "Not using GPU. Add the '-h' argument for available options.")
    else:
        logger.info("Not extracting features.")

    record_dirname = os.path.dirname(tfrecord_path)

    if not os.path.exists(record_dirname):
        os.makedirs(record_dirname)

    example_queue = multiprocessing.Queue(10)

    producer = multiprocessing.Process(
        target=single_consumer, args=(
            dataset, example_queue,
            extract_features,
            paralell_calls, gpu,
            batch_size))

    producer.start()

    writer = multiprocessing.Process(
        target=write, args=(
            example_queue,
            clean_data_dir,
            tfrecord_path,
            dataset))

    writer.start()

    producer.join()

    example_queue.put(None)
    writer.join()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Parse the dataset for efficient training.")

    parser.add_argument("--extract_features", action="store_true", help="""
    Use the gpu for improved performance. Depends on tensorflow-gpu.
    Olny used if extracting features.
    """)
    parser.add_argument("--gpu", action="store_true", help="""
    Use the gpu for improved performance. Depends on tensorflow-gpu.
    """)
    parser.add_argument("--paralell_calls", type=int, help="""
    The number of paralell calls to use for cpu map functions. 
    Defaults to 'None' (Means = 1).
    """)
    parser.add_argument("--batch_size", type=int, default=10, help="""
    The number of images that go through the deep learing model at once. 
    Large numbers can improve performance. Defaults to 10.
    """)
    parser.add_argument("--overwrite", action="store_true", help="""
    Overwrite an existing data.
    """)
    parser.add_argument("--parse_train", action="store_true", help="""
    Do we, or do we not parse the training images to tfrecord.
    """)
    parser.add_argument("--parse_predict", action="store_true", help="""
    Do we, or do we not parse the prediction images to tfrecord.
    """)
    parser.add_argument("--split_proba", type=float, default=0.8, help="""
    Probability of an example ending up in the train record. Defaults to 0.8.
    """)

    pathsJson = common.PathsJson()

    train_dataset = protein_dataset.Dataset(
        pathsJson.DIR_TRAIN,
        pathsJson.CSV_TRAIN)
    test_dataset = protein_dataset.Dataset(
        pathsJson.DIR_TEST,
        pathsJson.CSV_TEST)

    args = parser.parse_args()
    logger = logging.getLogger("ImageParser")

    paths = []
    if args.parse_train:
        logger.info("Parsing train.")
        paths.append(pathsJson.TRAIN_FEAURES_RECORD)
    if args.parse_predict:
        logger.info("Parsing predict.")
        paths.append(pathsJson.PREDICT_FEATURES_RECORD)

    for path in paths:
        if os.path.exists(path):
            if not args.overwrite:
                logger.error(
                    f"{path} exists. Use the flag '-h' for help.")
                exit(1)
            else:
                logger.warning(f"Overwriting {path}.")

    if args.parse_train:

        if not os.path.exists(pathsJson.TRAIN_DATA_CLEAN_PATH):
            os.makedirs(pathsJson.TRAIN_DATA_CLEAN_PATH)

        logger.info(f"Writing {pathsJson.TRAIN_FEAURES_RECORD}...")

        main(
            train_dataset,
            pathsJson.TRAIN_DATA_CLEAN_PATH,
            pathsJson.TRAIN_FEAURES_RECORD,
            args.paralell_calls,
            args.extract_features,
            args.gpu, args.batch_size)

    if args.parse_predict:

        if not os.path.exists(pathsJson.TEST_DATA_CLEAN_PATH):
            os.makedirs(pathsJson.TEST_DATA_CLEAN_PATH)

        logger.info(f"Writing {pathsJson.PREDICT_FEATURES_RECORD}...")
        main(
            test_dataset,
            pathsJson.TEST_DATA_CLEAN_PATH,
            pathsJson.PREDICT_FEATURES_RECORD,
            args.paralell_calls,
            args.extract_features,
            args.gpu, args.batch_size)

    splitrecords(
        pathsJson.TRAIN_FEAURES_RECORD,
        pathsJson.TRAIN_RECORD,
        pathsJson.TEST_RECORD,
        args.split_proba)
