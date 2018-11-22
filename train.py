import os
import shutil
import time
import threading
import logging
import random

import tensorflow as tf
import numpy as np

from package import common, dataset as dataset_module, model as model_module

def _model_fn(feature_tensor: tf.Tensor, label_tensor: tf.Tensor):

    model_config = common.TFHubModels(common.ConfigurationJson().TF_HUB_MODULE)
    model = model_module.ClassifierModel(True)

    feature_tensor.set_shape([None, model_config.feature_vector_size])
    logits_tensor = model.predict(feature_tensor)
    loss_tensor = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(label_tensor), logits=logits_tensor)
    loss_tensor = tf.reduce_mean(loss_tensor)
    pred_tensor = tf.nn.sigmoid(logits_tensor)

    return pred_tensor, loss_tensor, model

def _export_complete_model(export_model_dir: str):

    export_path = common.ConfigurationJson().EXPORTED_MODEL_DIR
    logger = logging.getLogger("ModelExporter")
    logger.info(f"Exporting to {export_path}")

    # pylint: disable=E1129
    with tf.Graph().as_default():
        sess = tf.Session()

        input_tensor = tf.placeholder(tf.uint8, [None, None, None, 3], name="input")

        feature_model = model_module.FeatureExractor(True)
        classifier = model_module.ClassifierModel()

        feature_map_tensor = feature_model.preprocess(input_tensor)
        feature_map_tensor = feature_model.predict(feature_map_tensor)

        logits = classifier.predict(feature_map_tensor)
        predictions_tensor = tf.sigmoid(logits, name="predictions")

        sess.run(tf.global_variables_initializer())
        
        for model in [feature_model, classifier]:
            try:
                model.load(sess)
            except ValueError:
                logger.warning(f"Restore failed. Loading default {model.variable_scope}")

        export_dir = common.ConfigurationJson().EXPORTED_MODEL_DIR

        if os.path.exists(export_dir):
            logger.warning(f"Overwriting {export_dir}...")
            shutil.rmtree(export_dir, True)

        tf.saved_model.simple_save(
            sess,
            common.ConfigurationJson().EXPORTED_MODEL_DIR,
            {"input": input_tensor},
            {"output": predictions_tensor})

        sess.close()

    logger.info("Done.")

def train(
        epochs: int,
        batch_size: int,
        train_backend: bool,
        cpu_only: bool,
        tfrecord: str):

    LR = 0.001

    logger = logging.getLogger("trainer")
    paths = common.PathsJson()

    logger.info(f"Using batch of {batch_size}")
    logger.info(f"Training for {epochs} epochs")
    if train_backend:
        logger.info(f"Training backend network")

    feature_batch = None
    label_batch = None
    models = []

    # pylint: disable=E1129
    with tf.Graph().as_default():

        dataset = tf.data.TFRecordDataset(tfrecord)
        dataset = dataset.repeat()

        feature_label_id_dataset = dataset.map(dataset_module.tf_parse_single_example,
            num_parallel_calls=os.cpu_count() + 1)

        if train_backend:

            feature_label_id_dataset = feature_label_id_dataset.map(
                lambda x, y, z: (x, y, dataset_module.tf_imgid_to_img_clean(
                    z, paths.TRAIN_DATA_CLEAN_PATH + "\\")), os.cpu_count())

        feature_batch, label_batch, img_batch = feature_label_id_dataset \
            .batch(batch_size) \
            .make_one_shot_iterator().get_next()

        if train_backend:

            model = model_module.FeatureExractor(True)
            feature_batch = model.predict(model.preprocess(img_batch))
            models.append(model)

        pred_tensor, loss_tensor, model = _model_fn(feature_batch, label_batch)
        optimize_op = tf.train.AdamOptimizer(LR).minimize(loss_tensor)
        
        models.append(model)

        config=tf.ConfigProto(
            device_count={'GPU': 0} if cpu_only else None,
        )

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for model in models:
                if model.checkpoint_available():
                    logger.info(f"Restoring {model.variable_scope}...")
                    model.load(sess)

            init = time.time()
            for i in range(epochs):
                _ = sess.run([optimize_op])

                if i % 100 == 0:
                    pr, lb, loss, _ = sess.run([pred_tensor, label_batch, loss_tensor, optimize_op])

                    pr = (pr > 0.5).astype(np.int32)
                    correct = np.sum(np.all(pr == lb, axis=1)) / np.size(pr, axis=0)

                    logger.info(f"Step: {i} of {epochs}, Loss: {loss}, correct: {correct}")
                
                if i % 1000 == 0 and i != 0:

                    end = time.time()
                    steps = 1000 * batch_size
                    timer = (end - init) / steps
                    timer *= 10**5

                    for model in models:
                        logger.info(f"Saving {model.variable_scope} for step {i}.")
                        model.save(sess)
                    logger.info(f"{round(timer, 4)}s / 10e5 examples.")

                    init = time.time()

            for model in models:
                logger.info(f"Finished training. Saving {model.variable_scope} to {paths.MODEL_CHECKPOINT_DIR}")
                model.save(sess)
            sess.close()


if __name__ == "__main__":

    import argparse
    import os

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_script")
    config = common.ConfigurationJson()
    paths = common.PathsJson()

    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(
        description="Parse the dataset for efficient training.")

    parser.add_argument("--overwrite", action="store_true", help="""
    Overwrite an existing saved_model directory.
    """)
    parser.add_argument("--train_feature_extractor", action="store_true", help="""
    Wether to train the backend CNN.
    """)
    parser.add_argument("--record", default=paths.TRAIN_RECORD, help=f"""
    Path of the training tfrecord. Defaults to {paths.TRAIN_RECORD}.
    """)
    parser.add_argument("--export_dir", type=str, default=config.EXPORTED_MODEL_DIR, help = f"""
    Final complete model export directory. Defaults to {config.EXPORTED_MODEL_DIR}.
    """)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help=f"""
    The number of training steps. Defaults to the config file value ({config.EPOCHS}).
    """)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help=f"""
    The number of images that go through the deep learing model at once.
    Large numbers can improve model quality. Defaults to the config file value ({config.BATCH_SIZE}).
    """)
    parser.add_argument("--cpu", action="store_true", help="""
    Force using the cpu even if a GPU is available.
    """)

    args = parser.parse_args()
    dirnames = [paths.MODEL_CHECKPOINT_DIR, args.export_dir]

    if args.overwrite:
        for dirname in dirnames:
            shutil.rmtree(dirname, True)

    train(args.epochs, args.batch_size, 
        args.train_feature_extractor, args.cpu, args.record)

    _export_complete_model(export_model_dir=args.export_dir)