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
            
            # NOTE (bcovas) Sanity check.
            # I fell for this one already.
            net = feature_tensor

            net = tf.nn.dropout(net, keep_prob)
            net = tf.layers.dense(net, 512, tf.nn.relu)
            net = tf.nn.dropout(net, keep_prob)
            net = tf.layers.dense(net, len(common.PROTEIN_LABEL.keys()), None)

            self._output = net

        return net



class ProteinEstimator(tf.estimator.Estimator):

    IMAGE_ID = "image_id"
    IMAGE_INPUT = "image_input"
    FEATURE_INPUT = "feature_input"
    HEAD_ONLY = "head_only"
    IMAGE_FEATURES = "image_features"
    SCORES = "scores"
    PREDICTIONS = "predictions"

    def __init__(self, config=None):
        super().__init__(
            model_fn=estimator_model_fn,
            model_dir=common.PathsJson().MODEL_CHECKPOINT_DIR,
            config=config)

def estimator_model_fn(features, labels, mode):

    head_only = False
    training = mode == tf.estimator.ModeKeys.TRAIN
    evaluating = mode == tf.estimator.ModeKeys.EVAL
    predicting = mode == tf.estimator.ModeKeys.PREDICT

    config = common.ConfigurationJson()
    model_config = common.TFHubModels(config.TF_HUB_MODULE)
    module = tf_hub.Module(model_config.url, trainable=training,
        tags={"train"} if training else None)

    img_id_tensor = features.get(dataset_module.TFRecordKeys.ID)
    img_tensor = features.get(dataset_module.TFRecordKeys.DECODED)
    feature_tensor = features.get(dataset_module.TFRecordKeys.IMG_FEATURES)

    if img_id_tensor is None:
        img_id_tensor = tf.convert_to_tensor("Null")
    if feature_tensor is not None:
        head_only = True
    if not head_only:
        feature_tensor = module(img_tensor)
        feature_tensor.set_shape([None, model_config.feature_vector_size])

    logits = ClassifierModel(training).predict(feature_tensor)

    prediction_scores = tf.nn.sigmoid(logits)
    predictions = tf.cast(prediction_scores > 0.5, tf.float32)

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

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=tf.to_float(labels))
    loss = tf.reduce_sum(loss)
    tf.summary.scalar("Loss", loss)

    if evaluating:

        metrics = {
            "Accuracy": tf.metrics.accuracy(labels, predictions),
            "FN": tf.metrics.false_negatives(labels, predictions),
            "FP": tf.metrics.false_positives(labels, predictions)
        }

        for name, metric in metrics.items():
            tf.summary.scalar(name, metric[1])
        tf.summary.image("ExampleImage", img_tensor, 8)

        return tf.estimator.EstimatorSpec(mode,
            predictions=predictions,
            eval_metric_ops=metrics,
            loss=loss)

    if training:

        metrics = {
            "Accuracy": tf.metrics.accuracy(labels, predictions)
        }

        for name, metric in metrics.items():
            tf.summary.scalar(name, metric[1])


        optimizer = tf.train.AdadeltaOptimizer(0.001)
        optimizer_op = optimizer.minimize(loss, tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode,
            predictions=predictions,
            loss=loss,
            train_op=optimizer_op,
            eval_metric_ops=metrics)

# NOT USED

class ExportedModel:

    _input_tensor = None
    _output_tensor = None

    def __init__(self):
        self.config = common.ConfigurationJson()

    def load(self, sess: tf.Session, input_tensor: tf.Tensor, model_dir=None):

        if model_dir is None:
            import glob
            files = glob.glob("**/*saved_model.pb*", recursive=True)
            model_dir = os.path.dirname(sorted(files)[-1])

        self._input_tensor = input_tensor

        graph_def = tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            model_dir,
            input_map={'input': input_tensor})

        outputs_mapping = dict(graph_def.signature_def['serving_default'].outputs)

        out_tensor_name = outputs_mapping['scores'].name
        self._output_tensor = tf.get_default_graph().get_tensor_by_name(out_tensor_name)
        
        return self._output_tensor

def export_model(dirname: str):

    estimator = ProteinEstimator()

    def input_function():

        example = tf.placeholder(tf.string, [], "example_placeholder")
        tensors_dict = dataset_module.tf_parse_single_example(example, [
                dataset_module.TFRecordKeys.ID,
                dataset_module.TFRecordKeys.IMG_PATHS,
                dataset_module.TFRecordKeys.IMG_FEATURES,
                dataset_module.TFRecordKeys.HEAD_ONLY
            ])

        tensors_dict[dataset_module.TFRecordKeys.DECODED] = \
            dataset_module.tf_load_image(
                tensors_dict[dataset_module.TFRecordKeys.IMG_PATHS])

        receiver = tf.estimator.export.ServingInputReceiver(
            tensors_dict,
            {"examples": example})

        return receiver

    estimator.export_saved_model(dirname, input_function)