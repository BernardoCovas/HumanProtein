import tensorflow as tf
import tensorflow_hub as tf_hub

from .common import ConfigurationJson, PathsJson, PROTEIN_LABEL

class FeatureExractor:

    def __init__(self):

        self.paths = PathsJson()
        self.config = ConfigurationJson()

        self._module = tf_hub.Module(self.config.tf_hub_module)

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

    def predict(self, feature_tensor: tf.Tensor):
        return self._model_fn(feature_tensor, None, False)

    def predict_train(self, feature_tensor: tf.Tensor, label_tensor: tf.Tensor):
        return self._model_fn(feature_tensor, label_tensor, True)

    def _model_fn(
            self,
            feature_tensor: tf.Tensor,
            label_tensor: tf.Tensor,
            is_training=False
        ):

        if is_training and label_tensor is None:
            return ValueError("Model is set to training but no labels were supplied.")

        self._input = feature_tensor

        net = tf.layers.dense(feature_tensor, 1024, tf.nn.relu)
        net = tf.layers.dense(feature_tensor, 512, tf.nn.relu)
        net = tf.layers.dense(feature_tensor, 256, tf.nn.relu)
        net = tf.layers.dense(feature_tensor, 128, tf.nn.relu)
        net = tf.layers.dense(
            feature_tensor, len(PROTEIN_LABEL.keys()), None)

        self._output = net

        if not is_training:
            return net

        loss = tf.losses.sigmoid_cross_entropy(label_tensor, net)
        return net, loss

    @property
    def input_tensor(self):
        return self._input

    @property
    def output_tensor(self):
        return self._output
