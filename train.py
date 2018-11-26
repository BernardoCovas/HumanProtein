import sys
import os
import shutil
import time
import logging
import random

import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np

from package import common, dataset as dataset_module, model as model_module

def export_model(dirname: str):

    logger = logging.getLogger("Exporter")
    logger.info(f"Exporting to {dirname}")

    tflogger = logging.getLogger("tensorflow")
    tflogger.propagate = False
    tflogger.setLevel(logging.INFO)

    estimator = tf.estimator.Estimator(
        model_fn=model_module.estimator_model_fn,
        model_dir=paths.MODEL_CHECKPOINT_DIR,
        config=None
    )

    def _inp_fn():

        img_paths = tf.placeholder(tf.string, [None, 4], name="input")
        imgs = tf.map_fn(lambda x: dataset_module.tf_load_image(x)[:, :, 0:3], img_paths, tf.uint8, 4)

        receiver = tf.estimator.export.ServingInputReceiver(
            {
                dataset_module.TFRecordKeys.DECODED_KEY: imgs
            },
            {
                "input": img_paths
            })

        return receiver

    estimator.export_saved_model(dirname, _inp_fn)

def train(
        clean: bool,
        epochs: int,
        batch_size: int,
        tfrecord_train: str,
        tfrecord_eval: str,
        eval_steps: int):

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)

    paths = common.PathsJson()
    config_json = common.ConfigurationJson()
    model_config = common.TFHubModels(config_json.TF_HUB_MODULE)

    logger.info(f"Using batch of {batch_size}")
    logger.info(f"Training for {epochs} epochs")
    logger.info(f"Running {eval_steps} evaluation steps")

    def _inp_fn(eval: bool):

        record = tfrecord_train
        if eval:
            record = tfrecord_eval

        dataset = tf.data.TFRecordDataset(record) \
                .shuffle(batch_size * 100)

        if not eval:
            dataset = dataset.repeat()

        def _map_fn(example):
            
            img_id, label = dataset_module.tf_parse_single_example(
                example,
                [
                    dataset_module.TFRecordKeys.ID_KEY,
                    dataset_module.TFRecordKeys.LABEL_KEY,
                ])

            if clean:
                data_path = os.path.normpath(paths.TRAIN_DATA_CLEAN_PATH)
                img = dataset_module.tf_imgid_to_img_clean(
                    img_id, data_path)
                return img, label

            data_path = os.path.normpath(paths.DIR_TRAIN)
            img = dataset_module.tf_imgid_to_img(
                img_id, data_path)[:, :, 0:3]

            img = tf.image.resize_bilinear([img], model_config.expected_image_size)[0]

            return {dataset_module.TFRecordKeys.DECODED_KEY: img}, label


        img_label = dataset.map(
            _map_fn, os.cpu_count() + 1) \
            .batch(batch_size) \
            .apply(tf.data.experimental.prefetch_to_device("gpu:0", 1))

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
    parser.add_argument("--export_only", action="store_true", help="""
    If you only wish to export the trained model.
    """)
    parser.add_argument("--clean", action="store_true", help="""
    Expect clean data. If you parsed the images in the parse_data.py script,
    use this flag.
    """)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help=f"""
    The number of images that go through the deep learing model at once.
    Large numbers can improve model quality. Defaults to the config file value ({config.BATCH_SIZE}).
    """)
    parser.add_argument("--eval_steps", type=int, default=1000, help=f"""
    Number of eval steps. Defaults to 1000.
    """)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help=f"""
    The number of training steps. Defaults to the config file value ({config.EPOCHS}).
    If -1, train indefinitely.
    """)
    parser.add_argument("--train_record", default=paths.TRAIN_RECORD, help=f"""
    Path of the training tfrecord. Defaults to {paths.TRAIN_RECORD}.
    """)
    parser.add_argument("--eval_record", default=paths.TEST_RECORD, help=f"""
    Path of the eval tfrecord. Defaults to {paths.TEST_RECORD}.
    """)
    parser.add_argument("--export_dir", type=str, default=config.EXPORTED_MODEL_DIR, help = f"""
    Final complete model export directory. Defaults to {config.EXPORTED_MODEL_DIR}.
    """)

    args = parser.parse_args()

    dirnames = [paths.MODEL_CHECKPOINT_DIR, args.export_dir]

    if args.overwrite:
        for dirname in dirnames:
            logger.warning(f"Overwriting: {dirname}")
            shutil.rmtree(dirname, True)

    for dirname in dirnames:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    if args.epochs < 0:
        args.epochs = None

    if not args.export_only:
        train(
            args.clean,
            args.epochs,
            args.batch_size,
            args.train_record,
            args.eval_record,
            args.eval_steps)

    export_model(args.export_dir)