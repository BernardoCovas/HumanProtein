import os
import logging

import tensorflow as tf
import tensorflow_hub as tf_hub

from .common import ConfigurationJson, PathsJson, PROTEIN_LABEL, TFHubModels

class ContainedModel:

    _scope="ExampleScope"
    _saver = None

    def __init__(self):
        self._model_dir = os.path.join(
            PathsJson().MODEL_CHECKPOINT_DIR, self._scope)
        self._logger = logging.getLogger(self._scope)

    @property
    def variable_scope(self):
        return self._scope

    def checkpoint_available(self):
        ch = tf.train.latest_checkpoint(self._model_dir)
        return ch is not None

    def load(self, sess: tf.Session):
        """
        Loads the saved_model variables. Returns the graph def.
        Run this AFTER the first predict, and AFTER tf.global_variables_initializer().
        (Or any other initializer that might overwrite the model variables)
        """

        self._io(sess, True)
        return self

    def save(self, sess: tf.Session):
        """
        Saves the model variables to the PathsJson's
        checkpoint folder.
        """
        self._io(sess, False)


    def _io(self, sess, restore: bool):

        paths = PathsJson()
        save_path = os.path.join(paths.MODEL_CHECKPOINT_DIR, self._scope, self._scope)
        
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        io_fn = self._saver.restore if restore else self._saver.save
        io_fn(sess, save_path)

    def _load_saver(self):

        var_list = tf.trainable_variables(scope=self.variable_scope)
        if len(var_list) == 0:
            self._logger.warning("No variables to save.")
            return
        self._saver = tf.train.Saver(var_list=var_list)

class FeatureExractor(ContainedModel):

    _scope = "FeaureExtractor"

    def __init__(self, trainable=False):
        super().__init__()
        
        self.paths = PathsJson()
        self.config = ConfigurationJson()
        self.trainable = trainable

        with tf.variable_scope(self._scope):
            self._module = tf_hub.Module(
                TFHubModels(self.config.TF_HUB_MODULE).url,
                trainable=trainable)

    @property
    def variable_scope(self):
        return self._scope

    def predict(self, image_tensor: tf.Tensor):
        """
        `image_tensor`: A preprocessed float32 4D tensor in [0, 1].
        """

        with tf.variable_scope(self._scope):
            out_tensor = self._module(image_tensor, self.trainable)
        self._load_saver()
        return out_tensor

    def preprocess(self, image_tensor: tf.Tensor, reshape=False):

        if reshape:

            height, width = tf_hub.get_expected_image_size(self._module)

            # NOTE (bcovas) image_tensor is a 4D tensor [batch, height, widt, channels]
            image_tensor = tf.image.resize_bilinear(
                image_tensor, [height, width])

        return image_tensor / 255

class ClassifierModel(ContainedModel):

    _input = None
    _output = None
    _scope = "ClassifierModel"

    def __init__(self, trainable=False):
        super().__init__()

        self._is_training = trainable
        self.paths = PathsJson()
        self.config = ConfigurationJson()

        self._module = tf_hub.Module(
            TFHubModels(self.config.TF_HUB_MODULE).url,
            trainable=trainable)

    def predict(self, feature_tensor: tf.Tensor):
        """
        Returns the classifier logits tensor.
        """ 
        res = self._model_fn(feature_tensor)
        self._load_saver()
        return res

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
            net = tf.layers.dense(net, 1024, tf.nn.relu)
            net = tf.nn.dropout(net, keep_prob)
            net = tf.layers.dense(net, len(PROTEIN_LABEL.keys()), None)

            self._output = net

        return net

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

