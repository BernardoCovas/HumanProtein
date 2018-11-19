import logging

from package.dataset import Dataset
from package.common import PathsJson

def main():
    dataset = Dataset()
    dataset.prepare()
    config = PathsJson()
    


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    main()
