import os
import threading
import logging
import datetime

import tensorflow as tf
import numpy as np

from package.model import ClassifierModel, FeatureExractor
from package.dataset import tf_parse_single_example, TFRecordKeys
from package.common import PROTEIN_LABEL, PathsJson, ConfigurationJson, TFHubModels

def _model_fn(feature_tensor: tf.Tensor, label_tensor: tf.Tensor):

    model_config = TFHubModels(ConfigurationJson().TF_HUB_MODULE)
    model = ClassifierModel()

    feature_tensor.set_shape([None, model_config.feature_vector_size])
    logits, loss = model.predict_train(feature_tensor, label_tensor)

    return logits, loss, model

def _export_complete_model(export_model_dir: str):

    export_path = ConfigurationJson().EXPORTED_MODEL_DIR
    logger = logging.getLogger("ModelExporter")
    logger.info(f"Exporting to {export_path}")

    # pylint: disable=E1129
    with tf.Graph().as_default():
        sess = tf.Session()

        input_tensor = tf.placeholder(tf.uint8, [None, None, None, 3], name="input")

        feature_model = FeatureExractor()
        classifier = ClassifierModel()

        feature_map_tensor = feature_model.preprocess(input_tensor)
        feature_map_tensor = feature_model.predict(feature_map_tensor)

        logits = classifier.predict(feature_map_tensor)
        predictions_tensor = tf.sigmoid(logits, name="predictions")

        sess.run(tf.global_variables_initializer())
        classifier.load(sess)

        tf.saved_model.simple_save(
            sess,
            ConfigurationJson().EXPORTED_MODEL_DIR,
            {"input": input_tensor},
            {"output": predictions_tensor})

        sess.close()

    logger.info("Done.")

def train(epochs: int, batch_size: int, prefetch: bool, cpu_only: bool):

    logger = logging.getLogger("trainer")
    paths = PathsJson()

    logger.info(f"Using batch of {batch_size}")
    logger.info(f"Training for {epochs} epochs")

    # pylint: disable=E1129
    with tf.Graph().as_default():

        dataset = tf.data.TFRecordDataset(PathsJson().TRAIN_DATA_CLEAN_PATH)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(os.cpu_count() + 1)
        dataset = dataset.map(tf_parse_single_example, num_parallel_calls=os.cpu_count() + 1)

        feature_dataset = dataset.map(lambda x: x[TFRecordKeys.IMG_FEATURES])
        label_dataset = dataset.map(lambda x: x[TFRecordKeys.LABEL_KEY])

        feature_dataset = feature_dataset.batch(batch_size)
        label_dataset = label_dataset.batch(batch_size)

        if prefetch:
            logger.warning(
"""
Using gpu prefetching. Notice that this op is saved in the graph, 
and might make the model unusable in cpu-only machines.
""")
            feature_dataset = feature_dataset.apply(tf.data.experimental.prefetch_to_device("gpu:0", 2))
            label_dataset   = label_dataset.apply(tf.data.experimental.prefetch_to_device("gpu:0", 2))

        else:
            feature_dataset = feature_dataset.prefetch(2)
            label_dataset = label_dataset.prefetch(2)

            feature_batch = feature_dataset.make_one_shot_iterator().get_next()
            label_batch   = label_dataset.make_one_shot_iterator().get_next()

        logits_tensor, loss_tensor, model = _model_fn(feature_batch, label_batch)
        optimize_op = tf.train.AdamOptimizer().minimize(loss_tensor)

        pred_tensor = tf.nn.sigmoid(logits_tensor)
        
        pred_mask = (pred_tensor == 1)
        label_mask = (label_batch == 1)
        all_maks = tf.logical_and(pred_mask, label_mask)
        accuracy_tensor = tf.reduce_sum(tf.to_float(all_maks)) / tf.size(all_maks, tf.float32)

        config=tf.ConfigProto(
            device_count={'GPU': 0} if cpu_only else None,
        )

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            for i in range(epochs):
                _ = sess.run([optimize_op])

                if i % 100 == 0:
                    pr, lb, loss, accuracy, _ = sess.run([pred_tensor, label_batch, loss_tensor, accuracy_tensor, optimize_op])
                    logger.info(f"Step: {i} of {epochs}, Loss: {loss}, Accuracy: {accuracy}")
                    print((pr[0] > 0.5).astype(np.int), '\n', lb[0])

            logger.info(f"Finished training. Saving model to {paths.MODEL_CHECKPOINT_DIR}")

            var_list = tf.get_default_graph().get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=model.variable_scope)

            saver = tf.train.Saver(var_list=var_list)
            saver.save(sess, paths.MODEL_CHECKPOINT_DIR)

            sess.close()


if __name__ == "__main__":

    import argparse
    import os
    import shutil

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("train_script")
    config = ConfigurationJson()
    paths = PathsJson()

    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(
        description="Parse the dataset for efficient training.")

    parser.add_argument("--overwrite", action="store_true", help="""
    Overwrite an existing saved_model directory.
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
    parser.add_argument("--prefetch_gpu", action="store_true", help="""
    Prefetch to gpu for improved performance. Saved model might not work on cpu-only machines.
    """)

    args = parser.parse_args()
    export_paths = [paths.MODEL_CHECKPOINT_DIR, args.export_dir]

    for path in export_paths:

        if os.path.exists(path):
            if args.overwrite:
                shutil.rmtree(path)
            else:
                logger.error(f"{path} not empty. Use '--help' for options.")
                exit()

    if args.prefetch_gpu and args.cpu:
        logger.error("Can't force cpu and prefetch gpu. See '--help'.")
        exit()

    train(args.epochs, args.batch_size, args.prefetch_gpu, args.cpu)
    _export_complete_model(export_model_dir=args.export_dir)