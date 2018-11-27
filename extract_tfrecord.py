#!/bin/python3

# AUTHOR b.covas (bernardo.covas@outlook.com)

# This file is meant for debugging.
# It extracts the contents of the tfrecord and prints them.
# It may provide some peace of mind if you are unsure about
# the tfrecord saving / unit testing.


import os
import random

import tensorflow as tf

from package.common import PathsJson
import package.dataset as dataset_module

def main(tfrecord_path: str):

    graph = tf.Graph()

    # pylint: disable=E1129
    with graph.as_default():

        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(lambda x: dataset_module.tf_parse_single_example(
            x, keys=[
                dataset_module.TFRecordKeys.ID,
                dataset_module.TFRecordKeys.LABEL,
                dataset_module.TFRecordKeys.IMG_PATHS]), 2)
        dataset = dataset.prefetch(None)

        features = dataset.make_one_shot_iterator().get_next()
        img_id_tensor = features[dataset_module.TFRecordKeys.ID]
        label_tensor = features[dataset_module.TFRecordKeys.LABEL]
        img_paths_tensor = features[dataset_module.TFRecordKeys.IMG_PATHS]

        sess = tf.Session()
        n = 0
        while True:
            try:
                img_id, labels, paths = sess.run([img_id_tensor, label_tensor, img_paths_tensor])
                print(f"{img_id.decode()} -> {str(labels)}")
                [print(p) for p in paths]
                n += 1
            except tf.errors.OutOfRangeError:
                break

        return n

if __name__ == "__main__":
    
    import argparse
    import logging
    logging.basicConfig(level=logging.INFO)
    pathsJson = PathsJson()

    parser = argparse.ArgumentParser(
        description="Extract and print tfrecord contents.")

    parser.add_argument("--test", action="store_true", help=f"""
    Do we parse test or train? Located at 
    {pathsJson.TEST_RECORD}
    {pathsJson.TRAIN_RECORD}
    """)

    parser.add_argument("--path", default=None, help=f"""
    Path to the tfrecord. Defaults to {pathsJson.TRAIN_RECORD}
    """)

    args = parser.parse_args()
    _path = args.path

    if _path is None:
        if args.test:
            _path = pathsJson.TEST_RECORD
        else:
            _path = pathsJson.TRAIN_RECORD

    n = main(_path)

    logging.getLogger("Extractor").info(f"""
    
    Extracted:
    {_path}
    With {n} examples.
    """)


