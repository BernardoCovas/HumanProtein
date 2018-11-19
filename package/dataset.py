import os
import shutil
import logging
import threading
import zipfile

import wget

class Dataset:
    """
    Base dataset class. Download the raw 'all.zip'
    from the competition website and place it in `dataset_path`.
    Calling `self.prepare()` will extract the dataset if not
    already extracted.

    `https://www.kaggle.com/c/human-protein-atlas-image-classification`
    """

    _logger = logging.getLogger("dataset_class")
    _filter_list = ["red", "green", "blue", "yellow"]
    # NOTE (bcovas) Almost for sure we do not need the yellow filter.

    dataset_path = ".data"
    _img_ids = []

    _all_zip = "all.zip"
    _train_zip = "train.zip"
    _test_zip = "test.zip"
    _csv_file = "train.csv"
    _csv_sample = "sample_submission.csv"
    _train_dir ="train"
    _test_dir = "test"

    _all_contents = [_train_dir, _test_dir, _csv_file, _csv_sample]

    def __init__(self, dataset_path="./.data"):
        self.dataset_path = dataset_path

    @property
    def train_dir(self):
        return self._raw_path(self._train_dir)

    @property
    def test_dir(self):
        return self._raw_path(self._test_dir)

    @property
    def ids(self):

        n_imgs = len(self._img_ids)
        if n_imgs == 0:
            raise IndexError(
                "Dataset not prepared. Don't forget to call self.prepare().")

        return self._img_ids

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
                for img_filter in self._filter_list:
                    imglist.append(_id + f"_{img_filter}" + ".png")

        return sorted(imglist) == sorted(existing_imglist)

# TENSORFLOW FUNCTIONS

def tf_write_single_example():
    pass

def tf_parse_single_example():
    pass