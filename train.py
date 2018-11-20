import logging

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

    return loss

def train(epochs: int, batch_size: int):

    logger = logging.getLogger("trainer")
    config = ConfigurationJson()

    if epochs is None:
        epochs = config.num_epochs
    if batch_size is None:
        batch_size = config.batch_size

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


        feature_dataset = feature_dataset.apply(tf.data.experimental.prefetch_to_device("gpu:0", batch_size))
        label_dataset   = label_dataset.apply(tf.data.experimental.prefetch_to_device("gpu:0", batch_size))

        loss_tensor = _model_fn(feature_dataset, label_dataset)
        optimize_op = tf.train.AdamOptimizer().minimize(loss_tensor)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(epochs):
                loss, _ = sess.run([loss_tensor, optimize_op])

                if i % 100 == 0:
                    logger.info(f"Step: {i} of {epochs}, Loss: {loss}")



if __name__ == "__main__":

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    logging.basicConfig(level=logging.INFO)

    train(None, None)