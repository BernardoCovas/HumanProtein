import os
import glob
import json

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

    def __init__(self, dirname: str, csv_file=None):
        """
        `dirname`: The directory containing all the images.
        `csv_file`: The associated csv file with ids and labels.
        """

        self._data = {}
        self._dirname = dirname
        self._csv_file = csv_file

        if csv_file is not None:
            self.reload()

    @property
    def img_ids(self):
        return list(self._data.keys())

    @property
    def img_id_and_paths(self):
        if not self.prepared():
            raise ValueError("Dataset seems to be currupted.")

        data = {}
        for img_id in self._data.keys():
            data[img_id] = self.get_img_paths(img_id)

        return data

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

    def scan_dir(self):

        img_ids = {}
        channels = []
        for channel in self.FILTER_LIST:
            files_list = glob.glob(os.path.join(self._dirname, f"*{channel}*"))
            channels.append(sorted(files_list))

        for img_channels in zip(*channels):
            img_id = os.path.commonprefix(img_channels)
            img_id = os.path.basename(img_id).replace("_", "")
            img_ids[img_id] = list(img_channels)

        return img_ids

    def get_img_paths(self, img_id: str) -> list:

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

    def label(self, img_id: str):
        return self._data.get(img_id)

    def label_vector(self, img_id: str):

        label_vector = np.zeros(
            [common.NUM_CLASSES], np.int)

        img_labels = self.label(img_id)
        img_labels = list(map(int, img_labels))

        for label in img_labels:
            label_vector[label] = 1

        return label_vector

    def vector_label(self, vector: np.ndarray):
        return common.one_hot_to_label(vector)

class PreProcessedDataset(Dataset):

    def get_img_path(self, img_id: str) -> str:

        if self._data.get(img_id) is None:
            raise ValueError(f"{img_id} does not exist in this dataset.")

        return os.path.join(self._dirname, f"{img_id}.png")


# TENSORFLOW FUNCTIONS

class TFRecordKeys:

    ID = "image/id"
    LABEL = "image/labels"
    IMG_PATHS = "image/paths"
    ENCODED = "image/encoded"
    DECODED = "image/decoded"
    IMG_FEATURES = "image/features"
    HEAD_ONLY = "model/head_only"

    KEYMAP = {
        ID: tf.FixedLenFeature([], tf.string, b"Null"),
        ENCODED: tf.FixedLenFeature([], tf.string),
        LABEL: tf.FixedLenSequenceFeature([], tf.int64, True),
        IMG_PATHS: tf.FixedLenSequenceFeature([], tf.string, True),
        IMG_FEATURES: tf.FixedLenSequenceFeature([], tf.float32, True),
        HEAD_ONLY: tf.FixedLenFeature([], tf.bool, False)
    }

def tf_load_clean_image():
    pass

def tf_load_image(paths: tf.Tensor, n_channels=3):

    channels = []
    for i in range(n_channels):
        img_bytes = tf.read_file(paths[i])
        img = tf.image.decode_image(img_bytes)
        channels.append(tf.squeeze(img))

    return tf.stack(channels, -1)

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

def tf_imgid_to_img_clean(img_id: str, dirname: str):
    """
    Utility function. Turns an image id into an image tensor.
    Does not join the diferent channels, expects clean images.

    NOTE: (bcovas) This might be temporary. Used for efficiency.
    """
    
    img_path = tf.string_join([dirname, img_id, ".png"])
    img_bytes = tf.read_file(img_path)
    img = tf.image.decode_image(img_bytes)

    return img

def tf_preprocessed_directory_dataset(dirname: str, paralell_readers=None):
    """
    Retruns a tf.data.Dataset that iterates `dirname` and yields 
    (image_tensor, img_fname).

    Does NOT apply any preprocessing except for joining channels.
    Matches EVERY file.
    """

    def _map_call(fname: str):
        img_bytes = tf.read_file(fname)
        return tf.image.decode_image(img_bytes)

    fname_dataset = tf.data.Dataset.list_files(dirname)
    img_dataset = fname_dataset.map(_map_call, paralell_readers)

    return tf.data.Dataset.zip((img_dataset, fname_dataset))


def tf_write_single_example(
    img_id: str, labels: [],
    img_paths : [],
    img_features: np.ndarray):

    feature = {
        TFRecordKeys.ID: _bytes_feature(img_id),
        TFRecordKeys.LABEL: _int64_feature(labels),
        TFRecordKeys.IMG_PATHS: _bytes_feature(img_paths),
        TFRecordKeys.IMG_FEATURES: _float_feature(img_features)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def tf_parse_single_example(
        serialized_example: bytes,
        keys=None):

    if keys is None:
        keys = [
            TFRecordKeys.LABEL,
            TFRecordKeys.ID,
        ]

    feature = {}
    for key in keys:
        feature[key] = TFRecordKeys.KEYMAP[key]

    features = tf.parse_single_example(serialized_example, features=feature)
    return features

def _int64_feature(value_list):
    if value_list is None:
        value_list = []
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))

def _float_feature(value):
    if value is None:
        value = []
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):

    if value is None:
        value = []
    if isinstance(value, str):
        value = value.encode()
    if not isinstance(value, list):
        value = [value]

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
