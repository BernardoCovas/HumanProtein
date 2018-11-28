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

def train(
        warm_start_dir: str,
        train_backend: bool,
        learning_rate: float,
        epochs: int,
        eval_steps: int,
        batch_size: int,
        train_record: str,
        eval_record: str,
        clean: bool,
        paralell_calls: int):

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)

    config_json = common.ConfigurationJson()
    model_config = common.TFHubModels(config_json.TF_HUB_MODULE)

    logger.info(("NOT " if not train_backend else "") + "Training backend")
    logger.info(f"Using learning rate of {learning_rate}")
    logger.info(f"Using batch of {batch_size}")
    logger.info(f"Training for {epochs} epochs")
    logger.info(f"Running {eval_steps} evaluation steps")

    def _inp_fn(eval: bool):
        crop = True

        record = train_record
        if eval:
            record = eval_record

        dataset = tf.data.TFRecordDataset(record) \
                .shuffle(batch_size * 100)

        if not eval:
            dataset = dataset.repeat()

        def _map_fn(example):
            
            features = dataset_module.tf_parse_single_example(
                example,
                [
                    dataset_module.TFRecordKeys.ID,
                    dataset_module.TFRecordKeys.LABEL,
                    dataset_module.TFRecordKeys.IMG_PATHS
                ])

            img_id = features[dataset_module.TFRecordKeys.ID]
            label = features[dataset_module.TFRecordKeys.LABEL]
            paths = features[dataset_module.TFRecordKeys.IMG_PATHS]

            img = dataset_module.tf_load_image(paths, n_channels=3)

            if crop:
                img = tf.random_crop(img, model_config.expected_image_size + (3,))
                img = tf.to_float(img)
            else:
                img = tf.image.resize_bilinear([img], model_config.expected_image_size)[0]

            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)

            return {
                dataset_module.TFRecordKeys.ID: img_id,
                dataset_module.TFRecordKeys.DECODED: img / 255,
            }, label

        features_label = dataset.apply(
            tf.data.experimental.map_and_batch(_map_fn, batch_size, paralell_calls)) \
            .prefetch(None) \

        return features_label

    # NOTE (bcovas) MirroredStrategy seems to be failing
    # on windows. Fails with no errors.
    strategy = tf_contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=1)
    config = tf.estimator.RunConfig(train_distribute=strategy)

    tflogger = logging.getLogger("tensorflow")
    tflogger.propagate = False

    estimator = model_module.ProteinEstimator(
        train_backend=train_backend,
        learning_rate=learning_rate,
        optimizer=tf.train.RMSPropOptimizer,
        warm_start_dir=warm_start_dir)

    train_spec = tf.estimator.TrainSpec(
        lambda: _inp_fn(False),
        max_steps=epochs)
    eval_spec = tf.estimator.EvalSpec(
        lambda: _inp_fn(True),
        steps=eval_steps)

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

    parser.add_argument("--override", action="store_true", help="""
    Overwrite an existing saved_model directory.
    """)
    parser.add_argument("--warm_start", action="store_true", help="""
    This flag needs to be set if the model changes somehow.
    (Ex: when we freeze the backend network and train the head with AdamOptimizer,
    and we want to unfreeze the backend network. Adam's vars will not be set for
    the backend network)
    """)
    parser.add_argument("--train_backend", action="store_true", help=f"""
    If set, train the backend network. Else just train the classifier head, keeping
    the rest of the network frozen.
    """)
    parser.add_argument("--learning_rate", type=float, default=0.01, help=f"""
    Learning rate to use in the training process. Defaults to 0.01
    """)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help=f"""
    The number of images that go through the deep learing model at once.
    Large numbers can improve model quality. Defaults to the config file value ({config.BATCH_SIZE}).
    """)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help=f"""
    The number of training steps. Defaults to the config file value ({config.EPOCHS}).
    If -1, train indefinitely.
    """)
    parser.add_argument("--eval_steps", type=int, default=1000, help=f"""
    Number of eval steps. Defaults to 1000.
    """)
    parser.add_argument("--train_record", default=paths.TRAIN_RECORD, help=f"""
    Path of the training tfrecord. Defaults to {paths.TRAIN_RECORD}.
    """)
    parser.add_argument("--eval_record", default=paths.TEST_RECORD, help=f"""
    Path of the eval tfrecord. Defaults to {paths.TEST_RECORD}.
    """)
    parser.add_argument("--clean", action="store_true", help="""
    Expect clean data. If you parsed the images in the parse_data.py script,
    use this flag.
    """)
    parser.add_argument("--export_dir", type=str, default=config.EXPORTED_MODEL_DIR, help = f"""
    Final complete model export directory. Defaults to {config.EXPORTED_MODEL_DIR}.
    """)

    args = parser.parse_args()
    old_checkpoint_dir = "models.old"

    dirnames = [
        paths.MODEL_CHECKPOINT_DIR,
        args.export_dir,
    ]

    if args.warm_start:
        shutil.rmtree(old_checkpoint_dir, True)
        # NOTE (bcovas) If tensorboard is active, this second step might fail.
        # By copying first and then deleting we ensure the dir is not lost.
        shutil.copytree(paths.MODEL_CHECKPOINT_DIR, old_checkpoint_dir)
        shutil.rmtree(paths.MODEL_CHECKPOINT_DIR, True)
        
    else:
        if args.override:
            for dirname in dirnames + [old_checkpoint_dir]:
                logger.warning(f"Overriding: {dirname}")
                shutil.rmtree(dirname, True)
        old_checkpoint_dir = None

    for dirname in dirnames:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    if args.epochs < 0:
        args.epochs = None

    train(
        old_checkpoint_dir,
        args.train_backend,
        args.learning_rate,
        args.epochs,
        args.eval_steps,
        args.batch_size,
        args.train_record,
        args.eval_record,
        args.clean,
        os.cpu_count() + 1)