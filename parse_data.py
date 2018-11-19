#!/bin/python

# This script is meant to parse the data.
# It uses the package.dataset class, and efficiently joins the
# different channels of the images.
# This script can be resource intensive.

import os
import queue
import threading
import multiprocessing
import logging

import tensorflow as tf
from tqdm import tqdm

from package.dataset import Dataset, tf_write_single_example
from package.common import PathsJson

def fname_loader():

    dataset = Dataset()
    dataset.prepare()

    for img_id in tqdm(dataset.train_ids):

        img_data = dataset.get_id_paths(img_id)
        yield img_data["red"], img_data["green"], img_data["blue"], img_data["labels"], img_id

def single_consumer(example_queue: queue.Queue, paralell_calls: int):

    if paralell_calls is None:
        paralell_calls = os.cpu_count()

    import logging
    logging.getLogger("image_parser").info(f"Using {paralell_calls} parallel calls.")

    def map_fn(r, g, b, lb, img_id):

        inputs = [r, g, b]
        channels = []

        for inp in inputs:
            file = tf.read_file(inp)
            
            channel = tf.image.decode_image(file, channels=1)
            channel = tf.squeeze(channel)
            channels.append(channel)

        image_tensor = tf.stack(channels, axis=-1)
        image_tensor = tf.image.encode_png(image_tensor)

        return image_tensor, lb, img_id

    import logging
    logging.basicConfig(level=logging.INFO)

    dataset = tf.data.Dataset.from_generator(fname_loader, 
        (tf.string, tf.string, tf.string, tf.int32, tf.string),
        (None, None, None, None, None))

    dataset = dataset.map(map_fn, paralell_calls)
    dataset = dataset.prefetch(5)

    # pylint: disable=E1129
    tf_graph = tf.Graph()
    with tf_graph.as_default():

        image_tensor, labels_tensor, img_id_tensor = dataset.make_one_shot_iterator().get_next()

        sess = tf.Session(config=tf.ConfigProto(
            device_count={'GPU': 0}
        ))

        while True:

            try:
                image_bytes, labels, img_id = sess.run([image_tensor, labels_tensor, img_id_tensor])
            except tf.errors.OutOfRangeError:
                break

            example_queue.put((image_bytes, labels, img_id))


def write(example_queue: queue.Queue, tfrecord_fname: str):

    with tf.python_io.TFRecordWriter(tfrecord_fname) as writer:

        while True:

            example = example_queue.get()
            if example is None:
                break

            image_bytes, labels, img_id = example

            serialized_example = tf_write_single_example(image_bytes, labels, img_id)
            writer.write(serialized_example)


def main(argv: []):

    logging.basicConfig(level=logging.INFO)

    tfrecord_fname = PathsJson().TRAIN_DATA_CLEAN_PATH

    dirname = os.path.dirname(tfrecord_fname)

    if not os.path.exists(dirname):
        os.makedirs(os.path.dirname(tfrecord_fname))

    example_queue = multiprocessing.Queue(100)

    producer = multiprocessing.Process(target=single_consumer, args=(example_queue, None))
    producer.start()

    writer = multiprocessing.Process(
        target=write, args=(example_queue, tfrecord_fname))
    writer.start()

    producer.join()

    example_queue.put(None)
    writer.join()


if __name__ == "__main__":

    main([])
