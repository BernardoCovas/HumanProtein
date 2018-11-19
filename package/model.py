import tensorflow as tf
import tensorflow_hub as tf_hub

class Model:

    TF_HUB_MODULE = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"

    def __init__(self):
        self._module = tf_hub.Module(self.TF_HUB_MODULE)

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