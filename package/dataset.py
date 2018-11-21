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

    _logger = logging.getLogger("DatasetClass")
    _filter_list = ["red", "green", "blue", "yellow"]
    # NOTE (bcovas) Almost for sure we do not need the yellow filter.

    dataset_path = ".data"
    _img_format = "png"
    _train_id_data = {}

    _all_zip = "all.zip"
    _train_zip = "train.zip"
    _test_zip = "test.zip"
    _train_csv = "train.csv"
    _sample_csv = "sample_submission.csv"
    _train_dir ="train"
    _test_dir = "test"

    _all_contents = [_train_dir, _test_dir, _train_csv, _sample_csv]
    _img_filters = ["red", "green", "blue", "yellow"]

    def __init__(self):
        self.dataset_path = common.PathsJson().RAW_DATA_DIR
        self._load_train_data()

    @property
    def train_dir(self):
        return self._raw_path(self._train_dir)

    @property
    def test_dir(self):
        return self._raw_path(self._test_dir)
    
    @property
    def num_examples(self):
        return len(self._train_id_data.keys())

    def prepare(self):

        if self._prepared():
            self._logger.info("Already extracted. Skipping.")
            return
        
        if not common.ConfigurationJson().OVERWRITE_DATASET_IF_CURRUPTED:
            self._logger.error("Dataset currupted.")
            exit(1)

        self._logger.info("Dataset currupted. Deleting...")

        # FIXME
        exit()

        shutil.rmtree(self._raw_path(self._train_dir), ignore_errors=True)
        shutil.rmtree(self._raw_path(self._test_dir), ignore_errors=True)

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


    def label(self, img_id: str):
        return self._train_id_data.get(img_id)

    def _load_train_data(self):
        
        # NOTE (bcovas) If already loaded
        if self._train_id_data != {}:
            return

        csv = self._raw_path(self._train_csv)
        with open(csv) as f:

            # NOTE (bcovas) Skipping header
            f.__next__()

            for row in f:
                row = row.replace("\n", "")
                img_id, label = row.split(",")
                img_labels = label.split(" ")
                self._train_id_data[img_id] = img_labels

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

        for _file in self._all_contents:

            _file = self._raw_path(_file)
            if not os.path.exists(_file):
                return False

        existing_imglist = os.listdir(self.train_dir)
        img_list = []

        for img_id in self._train_id_data.keys():
            for img_filter in self._img_filters:
                fname = f"{img_id}_{img_filter}.{self._img_format}"
                img_list.append(fname)

        return sorted(img_list) == sorted(existing_imglist)

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