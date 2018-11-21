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
from package.dataset import Dataset, tf_write_single_example, tf_preprocess_directory_dataset

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parse_data")


def single_consumer(
    dirname: str,
    example_queue: queue.Queue,
    paralell_calls: int,
    use_gpu: bool,
    batch_size=100,
    save_image=False):

    raw_dataset = Dataset()
    raw_dataset.prepare()

    if paralell_calls is None:
        paralell_calls = os.cpu_count()

    # pylint: disable=E1129
    tf_graph = tf.Graph()
    with tf_graph.as_default():

        tf_model = FeatureExractor()
        dataset = tf_preprocess_directory_dataset(
            dirname, paralell_calls)

        dataset = dataset.batch(batch_size)
        img_tensors, fnames_tensors = dataset.make_one_shot_iterator().get_next()
        preprocessed_img_tensors = tf_model.preprocess(img_tensors)
        features_tensors = tf_model.predict(preprocessed_img_tensors)

        sess = tf.Session(config=tf.ConfigProto(
            device_count={'GPU': 0 if not use_gpu else 1}
        ))

        sess.run(tf.global_variables_initializer())
        while True:
            try:
                example_queue.put(
                    sess.run([features_tensors, fnames_tensors]))
            except tf.errors.OutOfRangeError:
                break


def write(example_queue: queue.Queue, tfrecord_fname: str):

    logger = logging.getLogger("ExampleWriter")
    raw_dataset = Dataset()

    with tf.python_io.TFRecordWriter(tfrecord_fname) as writer:

        i = 0
        while True:

            example = example_queue.get()
            if example is None:
                break

            features, fnames = example

            for feature, fname in zip(features, fnames):

                one_hot = np.zeros([], np.int)

                img_id = strip_fname_for_id(fname[0].decode())
                img_labels = raw_dataset.label(img_id)
                img_labels = list(map(int, img_labels))

                for label in img_labels:
                    one_hot[label] = 1

                serialized_example = tf_write_single_example(
                    feature, one_hot, img_id)

                writer.write(serialized_example)

                if i % 100 == 0:
                    logger.info(
                        f"Wrote {i} examples")

                i += 1

    logger.info(f"Done. Wrote {i} examples.")


def main(
    dirname: str,
    tfrecord_fname: str, 
    n_paralell_calls: int, 
    gpu: bool, batch_size: 
    int, save_images: bool):

    """
    Even though, in this case, writing the features to disk 
    barely takes any time at all, I usually separate the 
    writing process from the prediction process.
    In some setups this might massively improve performance.
    """

    record_dirname = os.path.dirname(tfrecord_fname)

    if not os.path.exists(record_dirname):
        os.makedirs(record_dirname)

    example_queue = multiprocessing.Queue(10)

    producer = multiprocessing.Process(
        target=single_consumer, args=(
            dirname, example_queue,
            n_paralell_calls, gpu,
            batch_size, save_images))

    producer.start()

    writer = multiprocessing.Process(
        target=write, args=(example_queue, tfrecord_fname))
    writer.start()

    producer.join()

    example_queue.put(None)
    writer.join()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Parse the dataset for efficient training.")

    parser.add_argument("--save_images", action="store_true", help="""
    Save the images to the tfrecord. They are not needed for training, 
    and have a huge impact in performance. Defaults to 'False'.
    """)
    parser.add_argument("--gpu", action="store_true", help="""
    Use the gpu for improved performance. Depends on tensorflow-gpu.
    """)
    parser.add_argument("--n_paralell_calls", type=int, help="""
    The number of paralell calls to use for cpu map functions. 
    Defaults to 'None' (Means = number of cpus).
    """)
    parser.add_argument("--batch_size", type=int, default=50, help="""
    The number of images that go through the deep learing model at once. 
    Large numbers can improve performance. Defaults to 50.
    """)
    parser.add_argument("--overwrite", action="store_true", help="""
    Overwrite an existing tfrecord.
    """)
    parser.add_argument("--parse_eval", action="store_true", help="""
    Do we, or do we not parse the prediction images to tfrecord.
    """)

    args = parser.parse_args()

    logger = logging.getLogger("ImageParser")
    logger.info(f"Using batch_size of {args.batch_size}")
    logger.info(f"Using {args.n_paralell_calls} paralell calls.")

    train_tfrecord_fname = PathsJson().TRAIN_DATA_CLEAN_PATH
    test_tfrecord_fname = PathsJson().TEST_DATA_CLEAN_PATH

    if args.gpu:
        logger.info("Using GPU.")
    else:
        logger.warn(
            "Not using GPU. Add the '-h' argument for available options.")

    if args.save_images:
        logger.warning(
            "Saving images. This is not required for training and has a huge impact in performance.")
    else:
        logger.info("Not saving images, not required.")

    paths = [train_tfrecord_fname]
    if args.parse_eval:
        paths.append(test_tfrecord_fname)

    for path in paths:
        if os.path.exists(path):
            if not args.overwrite:
                logger.error(
                    f"{path} exists. Use the flag '-h' for help.")
                exit()
            else:
                logger.warning(f"Overwriting {path}.")

    logger.info(f"Parsing {train_tfrecord_fname}...")
#    main(
#        os.path.join(PathsJson().RAW_DATA_DIR, "train"), 
#        train_tfrecord_fname, args.n_paralell_calls, args.gpu,
#        args.batch_size, args.save_images)

    if args.parse_eval:
        logger.info(f"Parsing {test_tfrecord_fname}...")
        main(
            os.path.join(PathsJson().RAW_DATA_DIR, "test"),
            test_tfrecord_fname, args.n_paralell_calls, args.gpu,
            args.batch_size, args.save_images)
