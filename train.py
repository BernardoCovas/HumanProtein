import logging
import datetime

import tensorflow as tf

from package.model import ClassifierModel
from package.dataset import tf_parse_single_example, TFRecordKeys
from package.common import PROTEIN_LABEL, PathsJson, ConfigurationJson

def _model_fn(feature_dataset: tf.data.Dataset, label_dataset: tf.data.Dataset):

    model = ClassifierModel()

    feature_batch = feature_dataset.make_one_shot_iterator().get_next()
    label_batch   = label_dataset.make_one_shot_iterator().get_next()

    feature_batch.set_shape([None, 2048])

    _, loss = model.predict_train(feature_batch, label_batch)

    return loss, model.input_tensor, model.output_tensor

def train(epochs: int, batch_size: int, prefetch: bool):

    logger = logging.getLogger("trainer")
    paths = PathsJson()

    logger.info(f"Using batch of {batch_size}")
    logger.info(f"Training for {epochs} epochs")

    # pylint: disable=E1129
    with tf.Graph().as_default():

        dataset = tf.data.TFRecordDataset(PathsJson().TRAIN_DATA_CLEAN_PATH)
        dataset = dataset.repeat()
        dataset = dataset.map(tf_parse_single_example)

        feature_dataset = dataset.map(lambda x: x[TFRecordKeys.IMG_FEATURES])
        label_dataset = dataset.map(lambda x: x[TFRecordKeys.LABEL_KEY])

        feature_dataset = feature_dataset.batch(batch_size)
        label_dataset = label_dataset.batch(batch_size)

        # feature_dataset = feature_dataset.apply(tf.data.experimental.prefetch_to_device("gpu:0", batch_size))
        # label_dataset   = label_dataset.apply(tf.data.experimental.prefetch_to_device("gpu:0", batch_size))

        loss_tensor, model_inp_tensor, model_outp_tensor = _model_fn(feature_dataset, label_dataset)
        optimize_op = tf.train.AdamOptimizer().minimize(loss_tensor)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(epochs):
                loss, _ = sess.run([loss_tensor, optimize_op])

                if i % 100 == 0:
                    logger.info(f"Step: {i} of {epochs}, Loss: {loss}")

            logger.info(f"Finished training. Saving model to {paths.MODEL_CHECKPOINT_DIR}")
            tf.saved_model.simple_save(
                sess,
                paths.MODEL_CHECKPOINT_DIR,
                inputs={"input": model_inp_tensor},
                outputs={"output": model_outp_tensor})


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

    parser.add_argument("--prefetch_gpu", action="store_true", help="""
Prefetch to gpu for improved performance. Saved model might not work on cpu-only machines.
""")
    parser.add_argument("--epochs", type=int, default=config.num_epochs, help=f"""
The number of training steps. Defaults to the config file value ({config.num_epochs}).
""")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help=f"""
The number of images that go through the deep learing model at once. 
Large numbers can improve model quality. Defaults to the config file value ({config.batch_size}).
""")
    parser.add_argument("--overwrite", action="store_true", help="""
Overwrite an existing saved_model directory.
""")

    args = parser.parse_args()

    if os.path.exists(paths.MODEL_CHECKPOINT_DIR):
        if args.overwrite:
            shutil.rmtree(paths.MODEL_CHECKPOINT_DIR)
        else:
            logger.error("Checkpoint dir not empty. Use '-h' for options.")
            exit()

    train(args.epochs, args.batch_size, args.prefetch_gpu)