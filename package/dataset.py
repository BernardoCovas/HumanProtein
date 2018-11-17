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
    dataset_path = ".data"
    raw_zip_name = "all.zip"
    raw_zip_subfiles = ["test.zip", "train.zip"]

    def __init__(self, dataset_path="./.data"):
        self.dataset_path = dataset_path

    @property
    def raw_filename(self):
        return os.path.join(self.dataset_path, self.raw_zip_name)

    @property
    def train_dir(self):
        return os.path.join(self.dataset_path, "train")

    @property
    def test_dir(self):
        return os.path.join(self.dataset_path, "test")


    def prepare(self):

        self._extract(self.raw_filename)

        ts = []
        for filename in self.raw_zip_subfiles:

            t = threading.Thread(target=lambda: self._extract(os.path.join(self.dataset_path, filename)))
            ts.append(t)
            t.start()

        for t in ts:
            t.join()

    def _extract(self, filename: str):

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        self._logger.info(f"Extracting {filename}...")
        zip_file = zipfile.ZipFile(filename)

        dir_name = ""
        filelist = []
        zipfiles = []

        if filename == self.raw_filename:

            dir_name = self.dataset_path
            filelist = os.listdir(self.dataset_path)
            filelist.remove(self.raw_zip_name)
            zipfiles = zip_file.namelist()

        else:

            dir_name = filename.replace(".zip", "")
            filelist = os.listdir(dir_name) if os.path.exists(dir_name) else []
            zipfiles = zip_file.namelist()


        # NOTE (bcovas) We check if all the fles inside the
        # zip are extracted. If so, we skip.
        if set(sorted(zipfiles)) < set(sorted(filelist)):
            self._logger.info("Already extracted. Skipping...")
            return

        if os.path.exists(dir_name) and dir_name is not self.dataset_path:
            shutil.rmtree(dir_name)

        zip_file.extractall(dir_name)
        zip_file.close()

        self._logger.info(f"Done: {filename}")
