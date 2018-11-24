import os
import shutil
import time
import logging
import random

import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np

from package import common, dataset as dataset_module, model as model_module

def train(
        epochs: int,
        batch_size: int,
        tfrecord_train: str,
        tfrecord_eval: str,
        eval_steps: int):

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)

    paths = common.PathsJson()

    logger.info(f"Using batch of {batch_size}")
    logger.info(f"Training for {epochs} epochs")
    logger.info(f"Running {eval_steps} evaluatin steps")

    def _inp_fn(eval: bool):

        record = tfrecord_train
        if eval:
            record = tfrecord_eval

        dataset = tf.data.TFRecordDataset(record) \
                .repeat() \
                .shuffle(batch_size * 100)

        def _map_fn(example):

            img_id, label = dataset_module.tf_parse_single_example(
                example,
                [
                    dataset_module.TFRecordKeys.ID_KEY,
                    dataset_module.TFRecordKeys.LABEL_KEY
                ])

            img = dataset_module.tf_imgid_to_img_clean(
                img_id, paths.TRAIN_DATA_CLEAN_PATH + "\\")

            return img, label

        img_label = dataset.map(_map_fn, os.cpu_count() + 1) \
                .batch(batch_size)

        return img_label

    # NOTE (bcovas) MirroredStrategy seems to be failing
    # on windows. Fails with no errors.
    strategy = tf_contrib.distribute.MirroredStrategy(num_gpus=1)
    config = tf.estimator.RunConfig(train_distribute=strategy)

    estimator = tf.estimator.Estimator(
        model_fn=model_module.estimator_model_fn,
        model_dir=paths.MODEL_CHECKPOINT_DIR,
        config=None
    )

    train_spec = tf.estimator.TrainSpec(
        lambda: _inp_fn(False), 
        max_steps=epochs)
    eval_spec = tf.estimator.EvalSpec(
        lambda: _inp_fn(True),
        steps=eval_steps)

    tflogger = logging.getLogger("tensorflow")
    tflogger.propagate = False
    tflogger.setLevel(logging.INFO)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":

    import argparse
    import os

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger("train_script")
    logging.getLogger("tensorflow").setLevel(tf.logging.ERROR)

    config = common.ConfigurationJson()
    paths = common.PathsJson()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(
        description="Train the model on parsed data.")

    parser.add_argument("--overwrite", action="store_true", help="""
    Overwrite an existing saved_model directory.
    """)
    parser.add_argument("--train_record", default=paths.TRAIN_RECORD, help=f"""
    Path of the training tfrecord. Defaults to {paths.TRAIN_RECORD}.
    """)
    parser.add_argument("--eval_record", default=paths.TEST_RECORD, help=f"""
    Path of the eval tfrecord. Defaults to {paths.TEST_RECORD}.
    """)
    parser.add_argument("--eval_steps", type=int, default=100, help=f"""
    Number of eval steps. Defaults to 100. (For the entire test dataset)
    """)
    parser.add_argument("--export_dir", type=str, default=config.EXPORTED_MODEL_DIR, help = f"""
    Final complete model export directory. Defaults to {config.EXPORTED_MODEL_DIR}.
    """)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help=f"""
    The number of training steps. Defaults to the config file value ({config.EPOCHS}).
    If -1, train indefinitely.
    """)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help=f"""
    The number of images that go through the deep learing model at once.
    Large numbers can improve model quality. Defaults to the config file value ({config.BATCH_SIZE}).
    """)

    args = parser.parse_args()
    dirnames = [paths.MODEL_CHECKPOINT_DIR, args.export_dir]

    if args.overwrite:
        for dirname in dirnames:
            logger.warning(f"Overwriting: {dirname}")
            shutil.rmtree(dirname, True)

    if args.epochs < 0:
        args.epochs = None

    train(
        args.epochs,
        args.batch_size,
        args.train_record,
        args.eval_record,
        args.eval_steps)