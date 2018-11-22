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
    For configuration, use the provided `PATHS.json`.
    If your dir is already preprocessed, use 
    `PreProcessedDataset` instead.

    https://www.kaggle.com/c/human-protein-atlas-image-classification
    """

    FILTER_LIST = ["red", "green", "blue", "yellow"]
    _data = {}

    def __init__(self, dirname: str, csv_file: str):
        """
        `dirname`: The directory containing all the images.
        `csv_file`: The associated csv file with ids and labels.
        """

        self._dirname = dirname
        self._csv_file = csv_file
        self.reload()

    @property
    def img_ids(self):
        return list(self._data.keys())

    @property
    def csv_file(self):
        return self._csv_file

    @property
    def directory(self):
        return self._dirname
    
    def reload(self):
        """
        Reloads the associated csv file. This is generally
        not needed, but might come in handy if you are using
        and interactive python environment.
        """

        with open(self._csv_file) as f:

            # NOTE (bcovas) Skipping header
            f.__next__()

            for row in f:
                row = row.replace("\n", "")
                img_id, label = row.split(",")
                img_labels = label.split(" ")
                self._data[img_id] = img_labels

    def get_img_path(self, img_id: str) -> list:

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

            paths = self.get_img_path(img_id)
            for path in paths:
                if not os.path.exists(path):
                    return False

        return True

    def label(self, img_id: str):
        return self._data.get(img_id)


class PreProcessedDataset(Dataset):

    def get_img_path(self, img_id: str) -> str:

        if self._data.get(img_id) is None:
            raise ValueError(f"{img_id} does not exist in this dataset.")

        return os.path.join(self._dirname, f"{img_id}.png")


# TENSORFLOW FUNCTIONS

class TFRecordKeys:

    LABEL_KEY = "image/labels"
    ENCODED_KEY = "image/encoded"
    ID_KEY = "image/id"
    IMG_FEATURES = "image/features"

def tf_imgid_to_img(img_id: str, dirname: str):
    """
    Utility function. Turns an image id into an image tensor.
    Finds and joins the diferent channels.

    NOTE: (bcovas) This might be temporary. Used for efficiency.
    """
    channels = []
    for channel in Dataset.FILTER_LIST:
        image_bytes = tf.read_file(tf.string_join([
            dirname, "\\", img_id, "_", channel, ".png"]))
        channel = tf.image.decode_image(image_bytes)
        channels.append(tf.squeeze(channel))

    return tf.stack(channels, axis=-1)

def tf_preprocess_directory_dataset(dirname: str, paralell_readers=None):
    """
    Retruns a tf.data.Dataset that iterates `dirname` and yields 
    (image_tensor, img_fname).

    Does NOT apply any preprocessing except for joining channels.
    Matches EVERY file.
    """

    fname_dataset = tf.data.Dataset.list_files(dirname)
    img_bytes_dataset = fname_dataset.map(tf.read_file, paralell_readers)
    img_dataset = img_bytes_dataset.map(tf.image.decode_image, paralell_readers)

    return tf.data.Dataset.zip((img_dataset, fname_dataset))

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
