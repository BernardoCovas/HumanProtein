import os
import logging
import json

import numpy as np

PROTEIN_LABEL = {
    "0": "Nucleoplasm",
    "1": "Nuclear membrane",
    "2": "Nucleoli",
    "3": "Nucleoli fibrillar center",
    "4": "Nuclear speckles",
    "5": "Nuclear bodies",
    "6": "Endoplasmic reticulum",
    "7": "Golgi apparatus",
    "8": "Peroxisomes",
    "9": "Endosomes",
    "10": "Lysosomes",
    "11": "Intermediate filaments",
    "12": "Actin filaments",
    "13": "Focal adhesion sites",
    "14": "Microtubules",
    "15": "Microtubule ends",
    "16": "Cytokinetic bridge",
    "17": "Mitotic spindle",
    "18": "Microtubule organizing center",
    "19": "Centrosome",
    "20": "Lipid droplets",
    "21": "Plasma membrane",
    "22": "Cell junctions",
    "23": "Mitochondria",
    "24": "Aggresome",
    "25": "Cytosol",
    "26": "Cytoplasmic bodies",
    "27": "Rods & rings"
}

NUM_CLASSES = len(list(PROTEIN_LABEL.keys()))

class TFHubModel:

    @property
    def expected_image_size(self):
        raise NotImplementedError()
    
    @property
    def feature_vector_size(self):
        raise NotImplementedError()

    @property
    def url(self):
        raise NotImplementedError()

class PNASNet(TFHubModel):

    @property
    def expected_image_size(self):
        return (331, 331)

    @property
    def feature_vector_size(self):
        return 4320

    @property
    def url(self):
        return "https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/2"

class InceptionV3(TFHubModel):

    @property
    def expected_image_size(self):
        return (299, 299)

    @property
    def feature_vector_size(self):
        return 2048

    @property
    def url(self):
        return "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"

class MobileNetV2_140_224(TFHubModel):

    @property
    def expected_image_size(self):
        return (224, 224)

    @property
    def feature_vector_size(self):
        return 1792

    @property
    def url(self):
        return "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2"

class MobileNetV2_035_224(TFHubModel):

    @property
    def expected_image_size(self):
        return (224, 224)

    @property
    def feature_vector_size(self):
        return 1280

    @property
    def url(self):
        return "https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/2"

class ResNetV2_50(TFHubModel):

    @property
    def expected_image_size(self):
        return (224, 224)

    @property
    def feature_vector_size(self):
        return 2048

    @property
    def url(self):
        return "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1"

TFHUB_MODELS = {
    "PNASNet": PNASNet,
    "InceptionV3": InceptionV3,
    "MobileNetV2_140_224": MobileNetV2_140_224,
    "MobileNetV2_035_224": MobileNetV2_035_224,
    "ResNetV2_50": ResNetV2_50
}

def TFHubModels(model_name) -> TFHubModel:
    """
    Returns the corresponding TFHubModel.
    """

    logger = logging.getLogger("TFHubModels")
    model = TFHUB_MODELS.get(model_name)

    if model is None:
        logger.error(
f"""
Module name {model_name} not known.
Available:
{list(TFHUB_MODELS.keys())}
""")

    return model()

class ConfigurationJson:

    CONFIG_FNAME = "CONFIGURATION.json"
    EXAMPEL_CONFIG = {
        "TF_HUB_MODULE": "InceptionV3",
        "EXPORTED_MODEL_DIR": "./Exported/",
        "BATCH_SIZE": "100",
        "EPOCHS": "100000"
    }

    _config_data = {}

    def __init__(self):

        self._logger = logging.getLogger("ConfigJsonClass")

        if not os.path.exists(self.CONFIG_FNAME):

            self._logger.error(
    f"""

    No {self.CONFIG_FNAME} available.
    One will be created, please specify the locations for your config.
    """)

            with open(self.CONFIG_FNAME, "w") as json_file:
                json.dump(self.EXAMPEL_CONFIG, json_file, indent=2)
            exit(1)

        with open(self.CONFIG_FNAME) as json_file:
            self._config_data = json.load(json_file)

    @property
    def TF_HUB_MODULE(self):
        return str(self._config_data.get("TF_HUB_MODULE"))

    @property
    def BATCH_SIZE(self):
        return int(self._config_data.get("BATCH_SIZE"))

    @property
    def EPOCHS(self):
        return int(self._config_data.get("EPOCHS"))
    
    @property
    def EXPORTED_MODEL_DIR(self):
        return str(self._config_data.get("EXPORTED_MODEL_DIR"))

class PathsJson:

    _DIR_TRAIN = "train"
    _DIR_TEST = "test"
    _CSV_TRAIN = "train.csv"
    _CSV_TEST = "sample_submission.csv"
    _TRAIN_FEAURES_RECORD = "trainfeatures.record"
    _PREDICT_FEATURES_RECORD = "predfeatures.record"
    _TRAIN_RECORD = "trainrecord.record"
    _TEST_RECORD = "testrecord.record"

    PATHS_JSON_FNAME = "PATHS.json"
    EXAMPLE_PATHS_JSON = {
        "RAW_DATA_DIR": "./.data/",
        "TRAIN_DATA_CLEAN_PATH": "./.data/processed/train/",
        "TEST_DATA_CLEAN_PATH": "./.data/processed/test/",
        "MODEL_CHECKPOINT_DIR": "./models/",
        "LOGS_DIR": "./logs/",
        "SUBMISSION_DIR": "./submissions/"
    }

    _path_config = {}

    def __init__(self):

        self._logger = logging.getLogger("PathsJsonClass")
        
        if not os.path.exists(self.PATHS_JSON_FNAME):
        
            self._logger.error("""No PATHS.json available.
                One will be created, please specify the locations for your config.""")
            
            with open(self.PATHS_JSON_FNAME, "w") as json_file:
                json.dump(self.EXAMPLE_PATHS_JSON, json_file, indent=2)
            exit(1)

        with open(self.PATHS_JSON_FNAME) as json_file:
            self._path_config = json.load(json_file)

        for path in [
                        self.RAW_DATA_DIR,
                        self.TRAIN_DATA_CLEAN_PATH,
                        self.TEST_DATA_CLEAN_PATH
                    ]:
            if not os.path.exists(path) or len(os.listdir(path)) == 0:
                self._logger.warning(f"'{path}' is empty. Did you parse the data?")

    @property
    def RAW_DATA_DIR(self):
        return self._path_config.get("RAW_DATA_DIR")
    @property
    def TRAIN_DATA_CLEAN_PATH(self):
        return self._path_config.get("TRAIN_DATA_CLEAN_PATH")
    @property
    def TEST_DATA_CLEAN_PATH(self):
        return self._path_config.get("TEST_DATA_CLEAN_PATH")
    @property
    def MODEL_CHECKPOINT_DIR(self):
        return self._path_config.get("MODEL_CHECKPOINT_DIR")
    @property
    def RAW_DATLOGS_DIRA_DIR(self):
        return self._path_config.get("LOGS_DIR")
    @property
    def SUBMISSION_DIR(self):
        return self._path_config.get("SUBMISSION_DIR")
    @property
    def DIR_TRAIN(self):
        return os.path.join(self.RAW_DATA_DIR, self._DIR_TRAIN)
    @property
    def DIR_TEST(self):
        return os.path.join(self.RAW_DATA_DIR, self._DIR_TEST)
    @property
    def CSV_TRAIN(self):
        return os.path.join(self.RAW_DATA_DIR, self._CSV_TRAIN)
    @property
    def CSV_TEST(self):
        return os.path.join(self.RAW_DATA_DIR, self._CSV_TEST)
    @property
    def TRAIN_FEAURES_RECORD(self):
        return os.path.join(self.RAW_DATA_DIR, self._TRAIN_FEAURES_RECORD)
    @property
    def PREDICT_FEATURES_RECORD(self):
        return os.path.join(self.RAW_DATA_DIR, self._PREDICT_FEATURES_RECORD)
    @property
    def TRAIN_RECORD(self):
        return os.path.join(self.TRAIN_DATA_CLEAN_PATH, self._TRAIN_RECORD)
    @property
    def TEST_RECORD(self):
        return os.path.join(self.TEST_DATA_CLEAN_PATH, self._TEST_RECORD)

class Submission:

    HEADER = "Id,Predicted"

    def __init__(self, submission_file: str):

        self.submission_file = open(submission_file,"w")
        self.submission_file.write(self.HEADER + "\n")

    def add_submission(self, img_id: str, labels: list):

        labels_text = " ".join(labels)
        self.submission_file.write(",".join([img_id, labels_text]) + "\n")

    def end_sumbission(self):
        self.submission_file.close()

# UTILS

def one_hot_to_label(one_hot_list: np.ndarray):

    labels = np.arange(28)[one_hot_list == 1]
    return list(map(str, labels))

def strip_fname_for_id(fname: str):

    fname = os.path.basename(fname)
    img_id = fname.split("_")[0]
    return img_id