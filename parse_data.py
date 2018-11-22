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

from package.model import FeatureExractor
from package.common import PathsJson, strip_fname_for_id
from package import dataset as protein_dataset

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parse_data")


def single_consumer(
        dataset: protein_dataset.Dataset,
        example_queue: queue.Queue,
        paralell_calls: int,
        use_gpu: bool,
        batch_size: int,
        save_images: bool):

    # NOTE (bcovas) for some reason passing
    # this class to a process destroys it. 
    dataset.reload()

    # pylint: disable=E1129
    tf_graph = tf.Graph()
    with tf_graph.as_default():

        tf_model = FeatureExractor()

        ids_dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(dataset.img_ids, tf.string))
        img_dataset = ids_dataset.map(
            lambda x: protein_dataset.tf_imgid_to_img(x, dataset.directory)[:, :, 0:3],
            paralell_calls)
        img_bytes_dataset = img_dataset.map(
            tf.image.encode_png, paralell_calls)

        img_id_dataset = tf.data.Dataset.zip((
            img_dataset, ids_dataset, img_bytes_dataset)) \
            .prefetch(batch_size * 2) \
            .batch(batch_size)

        img_tensors, id_tensors, img_bytes_tensor = img_id_dataset.make_one_shot_iterator().get_next()

        preprocessed_img_tensors = tf_model.preprocess(img_tensors)
        features_tensors = tf_model.predict(preprocessed_img_tensors)

        sess = tf.Session(config=tf.ConfigProto(
            device_count={'GPU': 0 if not use_gpu else 1}
        ))

        sess.run(tf.global_variables_initializer())
        while True:
            try:
                if save_images:
                    example_queue.put(
                        sess.run([features_tensors, id_tensors, img_bytes_tensor]))
                else:
                    example_queue.put(
                        sess.run([features_tensors, id_tensors]))

            except tf.errors.OutOfRangeError:
                break

def write(
        example_queue: queue.Queue,
        clean_data_dir: str,
        tfrecord_fname: str,
        label_dataset: protein_dataset.Dataset,
        save_images: bool):

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

                one_hot = np.zeros([protein_dataset.common.NUM_CLASSES], np.int)
                img_labels = label_dataset.label(elements[1].decode())
                img_labels = list(map(int, img_labels))
                for label in img_labels:
                    one_hot[label] = 1

                serialized_example = protein_dataset.tf_write_single_example(
                    elements[0], one_hot, elements[1])
                writer.write(serialized_example)

                if save_images:

                    with open(
                            os.path.join(clean_data_dir, elements[1].decode() + ".png"),
                            "wb") as f:
                        f.write(elements[2])

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
        gpu: bool, batch_size:
        int, save_images: bool):
    """
    Even though, in this case, writing the features to disk 
    barely takes any time at all, I usually separate the 
    writing process from the prediction process.
    In some setups this might massively improve performance.
    """

    logger = logging.getLogger("MainImageParser")
    logger.info(f"Using batch_size of {batch_size}")
    logger.info(f"Using {paralell_calls} paralell calls.")
    if gpu:
        logger.info("Using GPU.")
    else:
        logger.warn(
            "Not using GPU. Add the '-h' argument for available options.")
    if save_images:
        logger.info(
            "Saving images.")
    else:
        logger.warning("Not saving images. " + \
            "You will not be able to train the feature extractor.")

    record_dirname = os.path.dirname(tfrecord_path)

    if not os.path.exists(record_dirname):
        os.makedirs(record_dirname)

    example_queue = multiprocessing.Queue(10)

    producer = multiprocessing.Process(
        target=single_consumer, args=(
            dataset, example_queue,
            paralell_calls, gpu,
            batch_size, save_images))

    producer.start()

    writer = multiprocessing.Process(
        target=write, args=(
            example_queue,
            clean_data_dir,
            tfrecord_path,
            dataset,
            save_images))

    writer.start()

    producer.join()

    example_queue.put(None)
    writer.join()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Parse the dataset for efficient training.")

    parser.add_argument("--no_images", action="store_false", help="""
    Don't save parsed images to disk.
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

    pathsJson = PathsJson()

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
            args.paralell_calls, args.gpu,
            args.batch_size, args.no_images)

    if args.parse_predict:

        if not os.path.exists(pathsJson.TEST_DATA_CLEAN_PATH):
            os.makedirs(pathsJson.TEST_DATA_CLEAN_PATH)

        logger.info(f"Writing {pathsJson.PREDICT_FEATURES_RECORD}...")
        main(
            test_dataset,
            pathsJson.TEST_DATA_CLEAN_PATH,
            pathsJson.PREDICT_FEATURES_RECORD,
            args.paralell_calls, args.gpu,
            args.batch_size, args.no_images)

    splitrecords(
        pathsJson.TRAIN_FEAURES_RECORD,
        pathsJson.TRAIN_RECORD,
        pathsJson.TEST_RECORD,
        args.split_proba)