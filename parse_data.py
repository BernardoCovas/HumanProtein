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
import queue
import threading
import multiprocessing
import logging
import itertools

import tensorflow as tf
import tensorflow_hub as tf_hub
import numpy as np

from package.dataset import Dataset, tf_write_single_example
from package.common import PathsJson
from package.model import Model

raw_dataset = Dataset()
raw_dataset.prepare()
total_examples = len(raw_dataset.train_ids)


def fname_loader():

    for img_id in raw_dataset.train_ids:

        img_data = raw_dataset.get_id_paths(img_id)
        yield img_data["red"], img_data["green"], img_data["blue"], img_data["labels"], img_id


def single_consumer(example_queue: queue.Queue, paralell_calls: int, use_gpu: bool, batch_size=50):

    if paralell_calls is None:
        paralell_calls = os.cpu_count()

    import logging
    logger = logging.getLogger("image_parser")
    logger.info(f"Using {paralell_calls} parallel calls.")

    msg = ""
    if use_gpu:
        msg = "Using GPU."
    else:
        msg = "Not using GPU. Add the '-h' argument for available options."

    logger.info(msg)

    def map_fn(r, g, b, lb, img_id):

        inputs = [r, g, b]
        channels = []

        for inp in inputs:
            file = tf.read_file(inp)

            channel = tf.image.decode_image(file, channels=1)
            channel = tf.squeeze(channel)
            channels.append(channel)

        image_tensor = tf.stack(channels, axis=-1)

        return image_tensor, lb, img_id

    import logging
    logging.basicConfig(level=logging.INFO)

    # pylint: disable=E1129
    tf_graph = tf.Graph()
    with tf_graph.as_default():

        tf_model = Model()

        dataset = tf.data.Dataset.from_generator(fname_loader,
                                                 (tf.string, tf.string,
                                                  tf.string, tf.int32, tf.string),
                                                 (None, None, None, None, None))

        dataset = dataset.map(map_fn, paralell_calls)

        img_dataset = dataset.map(lambda x, y, z: x)
        label_dataset = dataset.map(lambda x, y, z: y)
        id_dataset = dataset.map(lambda x, y, z: z)

        img_dataset = img_dataset.batch(batch_size)
        processed_img_dataset = img_dataset.map(tf_model.preprocess)

        if use_gpu:
            processed_img_dataset = processed_img_dataset.apply(
                tf.data.experimental.prefetch_to_device("/gpu:0"))

        label_dataset = label_dataset.padded_batch(
            batch_size, [tf.Dimension(None)], -1)
        id_dataset = id_dataset.batch(batch_size)

        processed_imgs_tensor = processed_img_dataset.make_one_shot_iterator().get_next()
        imgs_tensor = img_dataset.make_one_shot_iterator().get_next()
        labels_tensor = label_dataset.make_one_shot_iterator().get_next()
        img_ids_tensor = id_dataset.make_one_shot_iterator().get_next()

        features_tensor = tf_model.extract_features(processed_imgs_tensor)

        imgs_tensor = tf.map_fn(lambda x: tf.image.encode_png(
            x), imgs_tensor, dtype=tf.string)

        sess = tf.Session(config=tf.ConfigProto(
            device_count={'GPU': 0 if not use_gpu else 1}
        ))

        sess.run(tf.global_variables_initializer())
        while True:

            try:
                images_bytes, features, labels, img_ids = sess.run(
                    [imgs_tensor, features_tensor, labels_tensor, img_ids_tensor])
            except tf.errors.OutOfRangeError:
                break

            for i in range(labels.shape[0]):

                label = labels[i]
                label = label[label >= 0]

                example_queue.put(
                    (images_bytes[i], features[i], label, img_ids[i]))


def write(example_queue: queue.Queue, tfrecord_fname: str):

    logger = logging.getLogger("example_writer")

    with tf.python_io.TFRecordWriter(tfrecord_fname) as writer:
        for i in itertools.count():

            example = example_queue.get()
            if example is None:
                break
            if i % 100 == 0:
                logger.info("Wrote %i examples of %i" % (i, total_examples))

            image_bytes, features, labels, img_id = example

            serialized_example = tf_write_single_example(
                image_bytes, features, labels, img_id)
            writer.write(serialized_example)

    logger.info("Wrote all %i examples." % (total_examples))


def main(args):

    logging.basicConfig(level=logging.INFO)

    tfrecord_fname = PathsJson().TRAIN_DATA_CLEAN_PATH

    dirname = os.path.dirname(tfrecord_fname)

    if not os.path.exists(dirname):
        os.makedirs(os.path.dirname(tfrecord_fname))

    example_queue = queue.Queue(100)

    producer = threading.Thread(target=single_consumer, args=(
        example_queue, args.n_paralell_calls, args.gpu, args.batch_size))
    producer.start()

    writer = threading.Thread(
        target=write, args=(example_queue, tfrecord_fname))
    writer.start()

    producer.join()

    example_queue.put(None)
    writer.join()


if __name__ == "__main__":

    import argparse
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(
        description="Parse the dataset for efficient training.")

    parser.add_argument("--gpu", action="store_true", help="""
Use the gpu for improved performance. Depends on tensorflow-gpu.
""")
    parser.add_argument("--n_paralell_calls", type=int, help="""
The number of paralell calls to use for cpu map functions. Defaults to 'None' (Means = number of cpus).
""")
    parser.add_argument("--batch_size", type=int, default=50, help="""
The number of images that go through the deep learing model at once. 
Large numbers can improve performance. Defaults to 50.
""")

    args = parser.parse_args()
    main(args)
