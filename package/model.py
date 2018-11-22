import os

import tensorflow as tf
import tensorflow_hub as tf_hub

from .common import ConfigurationJson, PathsJson, PROTEIN_LABEL, TFHubModels

class FeatureExractor:

    def __init__(self):

        self.paths = PathsJson()
        self.config = ConfigurationJson()

        self._module = tf_hub.Module(TFHubModels(self.config.TF_HUB_MODULE).url)

    def predict(self, image_tensor: tf.Tensor):
        """
        `image_tensor`: A preprocessed float32 4D tensor in [0, 1].
        """
        return self._module(image_tensor)

    def preprocess(self, image_tensor: tf.Tensor):

        height, width = tf_hub.get_expected_image_size(self._module)

        # NOTE (bcovas) image_tensor is a 4D tensor [batch, height, widt, channels]
        image_tensor = tf.image.resize_bilinear(
            image_tensor, [height, width])

        return image_tensor / 255

class ClassifierModel:

    _input = None
    _output = None
    _scope = "ClassifierModel"

    def predict(self, feature_tensor: tf.Tensor):
        """
        Returns the classifier logits tensor.
        """
        return self._model_fn(feature_tensor, None, False)

    def predict_train(self, feature_tensor: tf.Tensor, label_tensor: tf.Tensor):
        return self._model_fn(feature_tensor, label_tensor, True)

    @property
    def variable_scope(self):
        return self._scope

    def _model_fn(
            self,
            feature_tensor: tf.Tensor,
            label_tensor: tf.Tensor,
            is_training=False
        ):

        if is_training:
            keep_prob = 0.5
        else:
            keep_prob = 1

        if is_training and label_tensor is None:
            return ValueError("Model is set to training but no labels were supplied.")

        with tf.variable_scope(self._scope):

            self._input = feature_tensor
            
            # NOTE (bcovas) Sanity check. I fell for this one already.
            net = feature_tensor

            net = tf.nn.dropout(net, keep_prob)
            net = tf.layers.dense(net, 1024, tf.nn.relu)
            net = tf.nn.dropout(net, keep_prob)
            net = tf.layers.dense(net, len(PROTEIN_LABEL.keys()), None)

            self._output = net

            if not is_training:
                return net

            loss = -tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(label_tensor), logits=net)
            
        return net, tf.reduce_mean(-loss)

    def load(self, sess: tf.Session):
        """
        Loads the saved_model variables. Returns the graph def.
        Run this AFTER the first predict, and AFTER tf.global_variables_initializer().
        (Or any other initializer that might overwrite the model variables)
        """

        paths = PathsJson()
        var_list = tf.trainable_variables(scope=self.variable_scope)
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, paths.MODEL_CHECKPOINT_DIR)


    @property
    def input_tensor(self):
        return self._input

    @property
    def output_tensor(self):
        return self._output

class ExportedModel:

    _input_tensor = None
    _output_tensor = None

    def __init__(self):
        self.config = ConfigurationJson()

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

