# You can run this script BEFORE starting the extraction.
# This detects the images that were extracted,
# converts them to a 3 channel jpg + a single jpg for the
# yellow channel and deletes the tiff images
# while the extraction is happening.
# It can also run after the extraction.

import os

import numpy as np
import cv2

from package import common, dataset as dataset_module

EXTENSION = "jpg"

def process(paths: [], out_file: str):
    imgs = []
    for path in paths:
        img = None
        while True:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to read {path}. Retrying.")
                continue
            break

        imgs.append(np.squeeze(img))

    img_rgb = np.stack(imgs[0:3], -1)
    img_yel = imgs[-1]

    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    cv2.imwrite(out_file, img_rgb)
    cv2.imwrite(out_file.replace(".", "_yellow."), img_yel)

def main(img_dataset: dataset_module.Dataset):

    if len(os.listdir(img_dataset.directory)) > 0:
        raise FileExistsError("Directory not empty.")

    unprocessed = img_dataset.img_ids
    processed = []

    print(f"Waiting for files in {img_dataset.directory}")

    i = 0
    n = len(unprocessed)
    while True:

        if len(processed) == len(unprocessed):
            print(f"Finished {n} examples.")
            break

        for img_id in unprocessed:

            paths = img_dataset.get_img_paths(img_id, "tif")
            ready = all(os.path.exists(p) for p in paths)

            if img_id not in processed and ready:
                i += 1
                process(paths, os.path.join(img_dataset.directory, f"{img_id}.{EXTENSION}"))
                processed.append(img_id)
                print(f"Wrote {img_id} ({i} of {n})")

                for path in paths:
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"Can't delete {path}: " + str(e))

if __name__ == "__main__":

    import argparse

    pathsJson = common.PathsJson()

    DIRS = {
        "test": (pathsJson.DIR_TEST, pathsJson.CSV_TEST),
        "train": (pathsJson.DIR_TRAIN, pathsJson.CSV_TRAIN)
    }

    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", required=True, help=f"""
    Directory to parse. One of {DIRS.keys()}
    """)

    args = parser.parse_args()
    dataset_args = DIRS.get(args.directory)

    if dataset_args is None:
        print(f"Unrecognized dir: {args.directory}")
        exit(1)

    dataset = dataset_module.Dataset(*dataset_args)
    main(dataset)