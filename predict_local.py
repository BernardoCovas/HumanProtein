import os
import time
import logging
import multiprocessing

import tensorflow as tf
import numpy as np

from package import common, model as model_module, dataset as dataset_module

def make_submission(batch_size: int, paralell_calls: int, sub_file: str):

    estimator = model_module.ProteinEstimator()
    model_config = common.TFHubModels(common.ConfigurationJson().TF_HUB_MODULE)

    dir_dataset = dataset_module.Dataset(common.PathsJson().DIR_TRAIN)
    img_ids_paths = dir_dataset.scan_dir()

    def g():
        for items in img_ids_paths.items():
            yield items

    def input_fn():
        def _map_fn(img_id, paths):

            img = dataset_module.tf_load_image(paths)
            img = tf.image.crop_to_bounding_box(
                img,
                tf.to_int32((tf.shape(img)[0] - model_config.expected_image_size[0]) / 2),
                tf.to_int32((tf.shape(img)[1] - model_config.expected_image_size[1]) / 2),
                model_config.expected_image_size[0],
                model_config.expected_image_size[1],
            )[:, :, 0:3] / 255

            return {
                dataset_module.TFRecordKeys.ID: img_id,
                dataset_module.TFRecordKeys.DECODED: img
            }

        dataset = tf.data.Dataset.from_generator(g, (tf.string, tf.string), ([], [None])) \
            .apply(tf.data.experimental.map_and_batch(_map_fn, batch_size)) \
            .prefetch(None)

        return dataset

    desired_pred= [
        estimator.IMAGE_ID,
        estimator.SCORES]

    submission = common.Submission(sub_file)
    logger = logging.getLogger("predictor")
    logger.info(f"Using batch of {batch_size}")
    logger.info(f"Using {paralell_calls} paralell calls")

    tf.logging.set_verbosity(tf.logging.INFO)
    for i, predictions in enumerate(estimator.predict(input_fn, desired_pred)):
        i += 1

        img_id = predictions[estimator.IMAGE_ID]
        scores = predictions[estimator.SCORES]
        labels = dir_dataset.vector_label((scores > 0.5).astype(np.int))
        submission.add_submission(img_id.decode(), labels)

        if i % 100 == 0:
            logger.info(f"Wrote {i} examples.")

    submission.end_sumbission()
    logger.info(f"Finished, wrote {i} examples.")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("tensorflow").propagate=False
    tf.logging.set_verbosity(tf.logging.ERROR)

    make_submission(10, 20, os.path.join(common.PathsJson().SUBMISSION_DIR, "prediction.csv"))