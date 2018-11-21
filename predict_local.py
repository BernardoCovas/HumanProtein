import os
import time
import logging
import multiprocessing

import tensorflow as tf
import numpy as np

from package.common import PathsJson, ConfigurationJson, one_hot_to_label, Submission
from package.model import ExportedModel
from package.dataset import tf_preprocess_directory_dataset

def run_prediction(queue, dirname: str, batch_size=50, paralell_calls=16):

    # pylint: disable=E1129
    with tf.Graph().as_default():
        sess = tf.Session()

        model = ExportedModel()
        dataset = tf_preprocess_directory_dataset(dirname, paralell_calls)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)

        image_tensor, img_id_tensor = dataset.make_one_shot_iterator().get_next()
        predict_tensor = model.load(sess, image_tensor)

        predict_tensor = predict_tensor > 0.5

        while True:
            try:
                preds, img_ids = sess.run([predict_tensor, img_id_tensor])
                queue.put((preds, img_ids))

            except tf.errors.OutOfRangeError:
                return

def make_submission(queue, submission_fname: str):

    logging.basicConfig(level=logging.INFO)

    submission = Submission(submission_fname)
    logger = logging.getLogger("Predictor")
    
    while True:
        
        preds_img_ids = queue.get()
        if preds_img_ids is None:
            break

        preds, img_ids = preds_img_ids
        labels = one_hot_to_label(preds)

        for img_id, label in list(zip(img_ids, labels)):
            
            img_id = img_id[0].decode()
            img_id = os.path.basename(img_id).replace("_red.png", "")

            submission.add_submission(img_id, label)

            logger.info(f"{img_id} -> {label}")

    submission.end_sumbission()
    logger.info("Done")


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)

    sub_file = "submission"
    paths = PathsJson()

    if not os.path.exists(paths.SUBMISSION_DIR):
        os.makedirs(paths.SUBMISSION_DIR)

    queue = multiprocessing.Queue(10)

    producer = multiprocessing.Process(target=run_prediction, args=(queue, paths.TEST_DATA_CLEAN_PATH))
    consumer = multiprocessing.Process(target=make_submission, args=(queue, sub_file))

    init = time.time()

    producer.start()
    consumer.start()

    producer.join()
    queue.put(None)
    consumer.join()

    end = time.time()
    print(end - init)
