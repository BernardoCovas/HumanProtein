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
    from the competition website and place it in `dataset_path`.
    Calling `self.prepare()` will extract the dataset if not
    already extracted.
    For configuration, use the provided `PATHS.json`.

    `https://www.kaggle.com/c/human-protein-atlas-image-classification`
    """

    _logger = logging.getLogger("dataset_class")
    _filter_list = ["red", "green", "blue", "yellow"]
    # NOTE (bcovas) Almost for sure we do not need the yellow filter.

    dataset_path = ".data"
    _train_img_ids = {}

    _all_zip = "all.zip"
    _train_zip = "train.zip"
    _test_zip = "test.zip"
    _csv_file = "train.csv"
    _csv_sample = "sample_submission.csv"
    _train_dir ="train"
    _test_dir = "test"

    _all_contents = [_train_dir, _test_dir, _csv_file, _csv_sample]

    def __init__(self):
        config = common.PathsJson()
        self.dataset_path = config.RAW_DATA_DIR

    @property
    def train_dir(self):
        return self._raw_path(self._train_dir)

    @property
    def test_dir(self):
        return self._raw_path(self._test_dir)

    @property
    def train_ids(self):

        n_imgs = len(self._train_img_ids.keys())
        if n_imgs == 0:
            raise IndexError(
                "Dataset not prepared. Don't forget to call self.prepare().")

        return self._train_img_ids

    def get_id_paths(self, img_id: str):

        if img_id not in self._train_img_ids.keys():
            raise IndexError("Not a valid id: " + img_id)

        data = {}
        for color_filter in self._filter_list:

            img_name = f"{img_id}_{color_filter}.png"
            img_name = os.path.join(self.dataset_path, self._train_dir, img_name)

            if not os.path.exists(img_name):
                raise FileNotFoundError(f"id {img_id} exists, but it's " + \
                    f"{color_filter} filter does not seem to exist.")

            data[color_filter] = img_name

        data["labels"] = self._train_img_ids.get(img_id)
        self._logger.debug("Found: " + img_id)

        return data

    def prepare(self):

        if self._prepared():
            self._logger.info("Already extracted. Skipping.")
            return

        _train_zip = self._raw_path(self._train_zip)
        _test_zip = self._raw_path(self._test_zip)

        self._extract(self._raw_path(self._all_zip), self.dataset_path)

        ts = []
        for f, d in list(zip([_train_zip, _test_zip], [self.train_dir, self.test_dir])):
            t = threading.Thread(target=self._extract, args=(f, d))
            ts.append(t)
            t.start()

        for t in ts:
            t.join()

        for f in [_train_zip, _test_zip]:
            self._logger.info(f"Deleting: {f}")
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

        self._logger.info("Done.")

    def _raw_path(self, file: str):
        return os.path.join(self.dataset_path, file)

    def _extract(self, filename: str, dirname=None):

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        self._logger.info(f"Extracting {filename}...")
        zip_file = zipfile.ZipFile(filename)

        zip_file.extractall(dirname)
        zip_file.close()

        self._logger.info(f"Done: {filename}")

    def _prepared(self):
        
        import csv

        for _file in self._all_contents:

            _file = self._raw_path(_file)
            if not os.path.exists(_file):
                return False

        existing_imglist = os.listdir(self.train_dir)
        imglist = []

        with open(self._raw_path(self._csv_file)) as f:
            reader = csv.DictReader(f)
            for row in reader:
                
                _id = row.get("Id")
                self._train_img_ids[_id] = row.get("Target").split(" ")

                for img_filter in self._filter_list:
                    imglist.append(_id + f"_{img_filter}" + ".png")

        return sorted(imglist) == sorted(existing_imglist)

# TENSORFLOW FUNCTIONS

class TFRecordKeys:

    LABEL_KEY = "image/labels"
    ENCODED_KEY = "image/encoded"
    ID_KEY = "image/id"
    IMG_FEATURES = "image/features"

def _int64_feature(value_list):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def tf_write_single_example(image: bytes, img_features: np.ndarray, labels: [], img_id: str):

    if type(img_id) == str:
        img_id = img_id.encode()

    feature = {TFRecordKeys.LABEL_KEY: _int64_feature(labels),
               TFRecordKeys.ENCODED_KEY: _bytes_feature(image),
               TFRecordKeys.ID_KEY: _bytes_feature(img_id),
               TFRecordKeys.IMG_FEATURES: _float_feature(img_features)}

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def tf_parse_single_example(serialized_example: bytes):

    feature = {TFRecordKeys.LABEL_KEY: tf.FixedLenSequenceFeature([], tf.int64,
                    allow_missing=True),
               TFRecordKeys.ENCODED_KEY: tf.FixedLenFeature([], tf.string),
               TFRecordKeys.ID_KEY: tf.FixedLenFeature([], tf.string),
               TFRecordKeys.IMG_FEATURES: tf.FixedLenSequenceFeature([], tf.float32,
                    allow_missing=True)
            }

    features = tf.parse_single_example(serialized_example, features=feature)

    return features