import logging

from package.dataset import Dataset


def main():
    dataset = Dataset()
    dataset.prepare()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    main()
