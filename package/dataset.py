import os
import shutil
import logging
import threading
import json
import zipfile

import wget
import tensorflow as tf
import numpy as np

from . import common

class Dataset:
    """
    Base dataset class. Download the raw 'all.zip'
    from the competition website.
    Calling `self.prepare()` will extract the dataset if not
    already extracted.
    For configuration, use the provided `PATHS.json`.

    `https://www.kaggle.com/c/human-protein-atlas-image-classification`
    """

    FILTER_LIST = ["red", "green", "blue", "yellow"]

    DIR_TRAIN = "train"
    DIR_TEST = "test"
    CSV_TRAIN = "train.csv"
    CSV_TEST = "sample_submission.csv"
    TFRECORD_TRAIN = "train.record"
    TFRECORD_TEST = "test.record"

    _data = {}

    def __init__(self, dirname: str, csv_file: str):
        self._dirname = dirname
        self._csv_file = csv_file
        self.load()

    @property
    def img_ids(self):
        return list(self._data.keys())
    
    def load(self):

        # NOTE (bcovas) If already loaded
        if self._data != {}:
            return

        with open(self._csv_file) as f:

            # NOTE (bcovas) Skipping header
            f.__next__()

            for row in f:
                row = row.replace("\n", "")
                img_id, label = row.split(",")
                img_labels = label.split(" ")
                self._data[img_id] = img_labels

    def get_img_paths(self, img_id: str):

        if self._data.get(img_id) is None:
            raise ValueError(f"{img_id} does not exist in this dataset.")

        paths = []
        for channel in self.FILTER_LIST:
            paths.append(os.path.join(self._dirname, f"{img_id}_{channel}.png"))

        return paths

    def prepared(self):

        if self._data == {}:
            return False

        for img_id in self._data.keys():

            paths = self.get_img_paths(img_id)
            for path in paths:
                if not os.path.exists(path):
                    return False

        return True

# TENSORFLOW FUNCTIONS

class TFRecordKeys:

    LABEL_KEY = "image/labels"
    ENCODED_KEY = "image/encoded"
    ID_KEY = "image/id"
    IMG_FEATURES = "image/features"

def tf_imgid_to_img(img_id: str, dirname: str):
    
    channels = []
    for channel in Dataset.FILTER_LIST:
        image_bytes = tf.read_file(tf.string_join([
            dirname, "\\", img_id, "_", channel, ".png"]))
        channel = tf.image.decode_image(image_bytes)
        channels.append(tf.squeeze(channel))


    return tf.stack(channels, axis=-1)

def tf_preprocess_directory_dataset(dirname: str, paralell_readers=None):
        """
        Retruns a tf.data.Dataset that iterates `dirname` and yields (image_tensor, img_id).
        """
        
        def _map_fn(*channels):

            image_tensor = []

            for channel in channels:
                channel = tf.image.decode_image(channel)
                channel = tf.squeeze(channel)
                
                image_tensor.append(channel)

            image_tensor = tf.stack(image_tensor, -1)

            return image_tensor
        
        filters = ["red", "green", "blue"]
        datasets = []
        file_name_datasets = []

        for filter_name in filters:
            dataset = tf.data.Dataset.list_files(
                # NOTE (bcovas) Match all files of a single filter
                os.path.join(dirname, f"*{filter_name}.png"), shuffle=False)
            file_name_datasets.append(dataset)
            dataset = dataset.map(lambda x: tf.read_file(x), paralell_readers)
            datasets.append(dataset)

        file_name_dataset = tf.data.Dataset.zip(tuple(file_name_datasets)) \
            .map(lambda *fnames: tf.stack(fnames))
        
        dataset = tf.data.Dataset.zip(tuple(datasets)).map(_map_fn, paralell_readers)

        return tf.data.Dataset.zip((dataset, file_name_dataset))

def _int64_feature(value_list):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    if type(value) == str:
        value = value.encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def tf_write_single_example(img_features: np.ndarray, labels: [], img_id: str, image_bytes=None):

    feature = {
        TFRecordKeys.LABEL_KEY: _int64_feature(labels),
        TFRecordKeys.ID_KEY: _bytes_feature(img_id),
        TFRecordKeys.IMG_FEATURES: _float_feature(img_features)
    }

    if image_bytes is not None:
        feature[TFRecordKeys.ENCODED_KEY] = _bytes_feature(image_bytes),

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def tf_parse_single_example(serialized_example: bytes, load_image_bytes=False):

    feature = {
        TFRecordKeys.IMG_FEATURES: tf.FixedLenSequenceFeature([], tf.float32,
            allow_missing=True),
        TFRecordKeys.LABEL_KEY: tf.FixedLenSequenceFeature([], tf.int64,
            allow_missing=True),
        TFRecordKeys.ID_KEY: tf.FixedLenFeature([], tf.string)
    }

    if load_image_bytes:
        feature[TFRecordKeys.ENCODED_KEY] = tf.FixedLenFeature([], tf.string)

    features = tf.parse_single_example(serialized_example, features=feature)
    return features
