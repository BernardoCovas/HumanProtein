import os
import logging

import tensorflow as tf
import tensorflow_hub as tf_hub

from . import common

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

        if self._is_training:
            keep_prob = 0.5
        else:
            keep_prob = 1

        with tf.variable_scope(self._scope):

            self._input = feature_tensor
            
            # NOTE (bcovas) Sanity check.
            # I fell for this one already.
            net = feature_tensor

            net = tf.nn.dropout(net, keep_prob)
            net = tf.layers.dense(net, 512, tf.nn.relu)
            net = tf.nn.dropout(net, keep_prob)
            net = tf.layers.dense(net, len(common.PROTEIN_LABEL.keys()), None)

            self._output = net

        return net

def estimator_model_fn(features, labels, mode):

    config = common.ConfigurationJson()
    tfmodel = common.TFHubModels(config.TF_HUB_MODULE)
    trainable = mode != tf.estimator.ModeKeys.EVAL

    module = tf_hub.Module(tfmodel.url, trainable=trainable)
    features.set_shape((None,) + tfmodel.expected_image_size + (3,))
    features = tf.to_float(features / 255)
    img_features = module(features, trainable)
    logits = ClassifierModel(trainable).predict(img_features)
    predictions = tf.nn.sigmoid(logits)
    predictions = tf.cast(predictions > 0.5, tf.float32)

    loss = None
    if trainable or mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=tf.to_float(labels))
        loss = tf.reduce_mean(loss)
        tf.summary.scalar("Loss", loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,
            predictions=predictions)
    if mode == tf.estimator.ModeKeys.EVAL:

        metrics = {
            "Accuracy": tf.metrics.accuracy(labels, predictions),
            "FN": tf.metrics.false_negatives(labels, predictions),
            "FP": tf.metrics.false_positives(labels, predictions)
        }

        for name, metric in metrics.items():
            tf.summary.scalar(name, metric[1])

        return tf.estimator.EstimatorSpec(mode,
            predictions=predictions,
            eval_metric_ops=metrics,
            loss=loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        metrics = {
            "Accuracy": tf.metrics.accuracy(labels, predictions)
        }

        for name, metric in metrics.items():
            tf.summary.scalar(name, metric[1])

        optimizer = tf.train.AdagradOptimizer(0.0001)
        optimizer_op = optimizer.minimize(loss, tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode,
            predictions=predictions,
            loss=loss,
            train_op=optimizer_op,
            eval_metric_ops=metrics)


class ExportedModel:

    _input_tensor = None
    _output_tensor = None

    def __init__(self):
        self.config = common.ConfigurationJson()

    def load(self, sess: tf.Session, input_tensor: tf.Tensor):

        self._input_tensor = input_tensor

        graph_def = tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            self.config.EXPORTED_MODEL_DIR,
            input_map={'input': input_tensor})

        outputs_mapping = dict(graph_def.signature_def['serving_default'].outputs)

        out_tensor_name = outputs_mapping['output'].name
        self._output_tensor = tf.get_default_graph().get_tensor_by_name(out_tensor_name)
        
        return self._output_tensor

