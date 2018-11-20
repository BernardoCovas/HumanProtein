import csv
import os

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

class ConfigurationJson:

    CONFIG_FNAME = "CONFIGURATION.json"
    EXAMPEL_CONFIG = {
        "TF_HUB_MODULE": "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1",
        "EXPORTED_MODEL_DIR": "./Exported/",
        "BATCH_SIZE": "100",
        "EPOCHS": "100000"
    }

    _config_data = {}

    def __init__(self):
        
        import os
        import logging
        import json

        self._logger = logging.getLogger("ConfigJsonClass")

        if not os.path.exists(self.CONFIG_FNAME):

            self._logger.error(
f"""
No {self.CONFIG_FNAME} available.
One will be created, please specify the locations for your config.
""")

            with open(self.CONFIG_FNAME, "w") as json_file:
                json.dump(self.EXAMPEL_CONFIG, json_file, indent=2)

            raise FileNotFoundError(f"No {self.CONFIG_FNAME} found.")

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

    PATHS_JSON_FNAME = "PATHS.json"
    EXAMPLE_PATHS_JSON = {
        "RAW_DATA_DIR": "./.data/",
        "TRAIN_DATA_CLEAN_PATH": "./.data/processed/train.record",
        "TEST_DATA_CLEAN_PATH": "./.data/processed/test.record",
        "MODEL_CHECKPOINT_DIR": "./models/",
        "LOGS_DIR": "./logs/",
        "SUBMISSION_DIR": "./submissions/"
    }

    _path_config = {}

    def __init__(self):

        import os
        import logging
        import json

        self._logger = logging.getLogger("PathsJsonClass")
        
        if not os.path.exists(self.PATHS_JSON_FNAME):
        
            self._logger.error("""No PATHS.json available.
                One will be created, please specify the locations for your config.""")
            
            with open(self.PATHS_JSON_FNAME, "w") as json_file:
                json.dump(self.EXAMPLE_PATHS_JSON, json_file, indent=2)

            raise FileNotFoundError("No PATHS.json found.")

        with open(self.PATHS_JSON_FNAME) as json_file:
            self._path_config = json.load(json_file)

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

class Submission:

    def __init__(self):

        self.paths = PathsJson()