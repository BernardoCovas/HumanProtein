#!/bin/python3

# This file is meant for debugging. 
# It extracts the contents of the tfrecord and prints them.

import os
import random

import tensorflow as tf

from package.common import PathsJson
import package.dataset as dataset_module

def main():

    train_record = PathsJson().TRAIN_DATA_CLEAN_PATH

    graph = tf.Graph()

    # pylint: disable=E1129
    with graph.as_default():

        dataset = tf.data.TFRecordDataset(train_record)
        dataset = dataset.map(dataset_module.tf_parse_single_example)

        features = dataset.make_one_shot_iterator().get_next()

        img_id_tensor = features[dataset_module.TFRecordKeys.ID_KEY]
        img_label_tensor = features[dataset_module.TFRecordKeys.LABEL_KEY]
        img_features_tensor = features[dataset_module.TFRecordKeys.IMG_FEATURES]

        sess = tf.Session()
        while True:
            try:
                img_id, labels, features = sess.run([img_id_tensor, img_label_tensor, img_features_tensor])
                print(f"{img_id.decode()} -> {str(labels)} : {len(features)} : {str(random.choice(features))[0:4]}")

            except tf.errors.OutOfRangeError:
                break

if __name__ == "__main__":
    main()