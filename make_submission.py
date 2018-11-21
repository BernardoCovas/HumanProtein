import os
import time
import logging
import multiprocessing

import tensorflow as tf
import numpy as np

from package.common import PathsJson, ConfigurationJson, one_hot_to_label, Submission
from package.model import ExportedModel
from package import dataset as protein_dataset

def _map_fn(*channels):

    image_tensor = []

    for channel in channels:
        channel = tf.image.decode_image(channel)
        channel = tf.squeeze(channel)
        
        image_tensor.append(channel)

    image_tensor = tf.stack(image_tensor, -1)

    return image_tensor

def run_prediction(queue, ids: [], dirname: str, batch_size: int, paralell_calls: int):

    # pylint: disable=E1129
    with tf.Graph().as_default():
        sess = tf.Session()
        ids = tf.constant(ids, tf.string)
        ids_dataset = tf.data.Dataset.from_tensor_slices(ids)
        img_dataset = ids_dataset.map(
            lambda x: protein_dataset.tf_imgid_to_img(x, dirname), paralell_calls)

        dataset = tf.data.Dataset.zip((img_dataset, ids_dataset))

        model = ExportedModel()

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)

        image_tensor, img_id_tensor = dataset.make_one_shot_iterator().get_next()
        predict_tensor = model.load(sess, image_tensor[:, :, :, 0:3])

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

    paths = PathsJson()
    sub_file = os.path.join(paths.SUBMISSION_DIR, "submission.csv")

    if not os.path.exists(paths.SUBMISSION_DIR):
        os.makedirs(paths.SUBMISSION_DIR)

    queue = multiprocessing.Queue(10)
    dataset = protein_dataset.Dataset(
        os.path.join(paths.RAW_DATA_DIR, protein_dataset.Dataset.DIR_TEST),
        os.path.join(paths.RAW_DATA_DIR, protein_dataset.Dataset.CSV_TEST))

    producer = multiprocessing.Process(target=run_prediction, args=(
        queue, 
        protein_dataset.Dataset(paths.TEST_DATA_CLEAN_PATH,
            protein_dataset.Dataset.CSV_TEST).img_ids,
        os.path.join(paths.RAW_DATA_DIR, protein_dataset.Dataset.DIR_TEST)
        , 1, 16))

    consumer = multiprocessing.Process(target=make_submission, args=(queue, sub_file))

    init = time.time()

    producer.start()
    consumer.start()

    producer.join()
    queue.put(None)
    consumer.join()

    end = time.time()
    print(end - init)
