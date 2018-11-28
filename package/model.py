import os
import logging

import tensorflow as tf
import tensorflow_hub as tf_hub

from . import common, dataset as dataset_module


class ClassifierModel:

    _input = None
    _output = None
    _scope = "ClassifierModel"

    def __init__(self, trainable=False):
        self._is_training = trainable

    def predict(self, feature_tensor: tf.Tensor):
        """
        Returns the classifier logits tensor.
        """ 
        return self._model_fn(feature_tensor)

    @property
    def input_tensor(self):
        return self._input

    @property
    def output_tensor(self):
        return self._output

    def _model_fn(
            self,
            feature_tensor: tf.Tensor,
        ):

        with tf.variable_scope(self._scope):

            self._input = feature_tensor
            keep_prob = 1
            if self._is_training:
                keep_prob = 0.5

            hidden = [512]
            
            # NOTE (bcovas) Sanity check.
            # I fell for this one already.
            net = feature_tensor
            for n in hidden:
                net = tf.nn.dropout(net, keep_prob)
                net = tf.layers.dense(net, n, tf.nn.relu)

            net = tf.nn.dropout(net, keep_prob)
            net = tf.layers.dense(net, len(common.PROTEIN_LABEL.keys()), None)

            self._output = net

        return net



class ProteinEstimator(tf.estimator.Estimator):
    """
    Custom estimator class. This class contains Keys
    that might be used in the model_fn.
    If you need fast prototyping, change it's model_funtion
    (`estimator_model_fn`) in package/model.py.

    Information on how this function works can be found at:
    https://www.tensorflow.org/guide/custom_estimators#write_a_model_function
    """

    IMAGE_ID = "image_id"
    IMAGE_INPUT = "image_input"
    FEATURE_INPUT = "feature_input"
    HEAD_ONLY = "head_only"
    IMAGE_FEATURES = "image_features"
    SCORES = "scores"
    PREDICTIONS = "predictions"

    def __init__(self,
        model_dir=None,
        warm_start_dir=None,
        config=None,
        train_backend=False,
        optimizer=tf.train.AdamOptimizer,
        learning_rate=0.001):

        if model_dir is None:
            model_dir = common.PathsJson().MODEL_CHECKPOINT_DIR

        def _model_fn(features, labels, mode, config):
            return estimator_model_fn(
                features, labels, mode, config,
                train_backend, learning_rate,
                optimizer, model_dir)

        super().__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            warm_start_from=warm_start_dir)

def estimator_model_fn(
        features,
        labels,
        mode,
        config: tf.estimator.RunConfig,
        train_backend: bool,
        learning_rate: float,
        tf_optimizer: tf.train.Optimizer,
        model_dir: str):

    training = mode == tf.estimator.ModeKeys.TRAIN
    evaluating = mode == tf.estimator.ModeKeys.EVAL
    predicting = mode == tf.estimator.ModeKeys.PREDICT

    head_only = False
    train_backend = training and train_backend

    config = common.ConfigurationJson()
    model_config = common.TFHubModels(config.TF_HUB_MODULE)
    module = tf_hub.Module(model_config.url, trainable=True,
        tags=["train"] if train_backend else None)

    img_id_tensor = features.get(dataset_module.TFRecordKeys.ID)
    img_tensor = features.get(dataset_module.TFRecordKeys.DECODED)
    feature_tensor = features.get(dataset_module.TFRecordKeys.IMG_FEATURES)

    if img_id_tensor is None:
        img_id_tensor = tf.convert_to_tensor(b"Null")
    if feature_tensor is not None:
        head_only = True
    if not head_only:
        feature_tensor = module(img_tensor)
        feature_tensor.set_shape([None, model_config.feature_vector_size])
        if not train_backend:
            feature_tensor = tf.stop_gradient(feature_tensor)

    logits = ClassifierModel(training).predict(feature_tensor)

    prediction_scores = tf.nn.sigmoid(logits)
    predictions = tf.cast(prediction_scores > 0.5, tf.int64)

    if predicting:

        prediction_dict = {
            ProteinEstimator.SCORES: prediction_scores,
            ProteinEstimator.PREDICTIONS: predictions,
            ProteinEstimator.IMAGE_ID: img_id_tensor,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: predictions
        }

        if not head_only:
            prediction_dict[ProteinEstimator.IMAGE_FEATURES] = feature_tensor

        return tf.estimator.EstimatorSpec(mode,
            predictions = prediction_dict)

    tf.losses.sigmoid_cross_entropy(tf.to_float(labels), logits)
    total_loss = tf.losses.get_total_loss()

    correct = tf.reduce_all(tf.equal(labels, predictions), axis=1)
    correct = tf.cast(correct, tf.int32)
    correct = correct / tf.shape(correct)
    correct = tf.reduce_sum(correct)

    FN = tf.reduce_sum(
        tf.cast(tf.logical_and(tf.equal(labels, 1), tf.equal(predictions , 0)), tf.int32))
    FP = tf.reduce_sum(
        tf.cast(tf.logical_and(tf.equal(labels, 0), tf.equal(predictions , 1)), tf.int32))

    FN = FN / tf.size(labels)
    FP = FP / tf.size(labels)

    if evaluating:

        metrics = {
            "Correct": tf.metrics.mean(correct),
            "FN": tf.metrics.mean(FN),
            "FP": tf.metrics.mean(FP),
            "Accuracy": tf.metrics.accuracy(labels, predictions),
        }

        return tf.estimator.EstimatorSpec(mode,
            predictions=predictions,
            eval_metric_ops=metrics,
            loss=total_loss)

    if training:

        if not head_only:
            tf.summary.image("ExampleImage", img_tensor[:1], 1)
        tf.summary.scalar("Correct", correct)
        tf.summary.scalar("FN", FN)
        tf.summary.scalar("FP", FP)

        optimizer: tf.train.Optimizer = tf_optimizer(learning_rate)
        optimizer_op = optimizer.minimize(total_loss, tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode,
            predictions=predictions,
            loss=total_loss,
            train_op=optimizer_op)

def focal_loss(predictions, labels, gamma=2.):
    """
    This implementation is experimental, and numerically unstable.
    Might cause NaN's during training.
    """

    max_val = tf.maximum(-predictions, 0)
    loss = predictions - predictions * labels + max_val + \
        tf.math.log(tf.math.exp(-max_val) + tf.math.exp(-predictions - max_val))

    invprops = tf.log_sigmoid(-predictions * (labels * 2 - 1))
    loss = tf.math.exp(invprops * gamma) * loss

    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
