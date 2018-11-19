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