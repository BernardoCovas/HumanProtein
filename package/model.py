import tensorflow as tf
import tensorflow_hub as tf_hub

from .common import ConfigurationJson

class Model:

    def __init__(self):
        self.config = ConfigurationJson()
        self._module = tf_hub.Module(self.config.tf_hub_module)

    def predict(self, image: []):
        pass

    def preprocess(self, images_tensor: tf.Tensor, normalize=True):

        height, width = tf_hub.get_expected_image_size(self._module)

        # NOTE (bcovas) image_tensor is a 4D tensor [batch, height, widt, channels]
        images_tensor = tf.image.resize_bilinear(images_tensor, [height, width])

        if normalize:
            images_tensor = images_tensor / 255
        
        return images_tensor

    def extract_features(self, images_tensor: tf.Tensor, normalize=True):
        features = self._module(images_tensor)
        return features