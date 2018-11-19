#!/bin/python3

# This file is meant for debugging. It extracts the contents of the tfrecord
# and saves every image as $(img_id)($(labels)).png,
# example: 32424qfqwfq34r(12_1).png.

import os

import tensorflow as tf

from package.common import PathsJson
import package.dataset as dataset_module

def main():

    train_record = PathsJson().TRAIN_DATA_CLEAN_PATH
    out_path = os.path.dirname(train_record) + "/"

    graph = tf.Graph()

    # pylint: disable=E1129
    with graph.as_default():

        dataset = tf.data.TFRecordDataset(train_record)
        dataset = dataset.map(dataset_module.tf_parse_single_example)

        features = dataset.make_one_shot_iterator().get_next()

        # img_features_tensor = features[dataset_module.TFRecordKeys.IMG_FEATURES]
        image_bytes_tensor = features[dataset_module.TFRecordKeys.ENCODED_KEY]
        img_tensor = tf.image.decode_image(image_bytes_tensor)
        shape_tensor = tf.shape(img_tensor)

        img_id_tensor = features[dataset_module.TFRecordKeys.ID_KEY]
        img_label_tensor = features[dataset_module.TFRecordKeys.LABEL_KEY]

        out_path_tensor = out_path + features[dataset_module.TFRecordKeys.ID_KEY] + \
            "(" + tf.strings.reduce_join(
                tf.as_string(img_label_tensor), separator="_") + ").png"
        
        write_file = tf.write_file(out_path_tensor, image_bytes_tensor)

        sess = tf.Session()
        while True:
            try:
                _, img_id, shape, labels = sess.run([write_file, img_id_tensor, shape_tensor, img_label_tensor])
                print("Wrote: " + img_id.decode() + " of shape: " + str(shape) + " and labels: " + str(labels))

            except tf.errors.OutOfRangeError:
                print("Done.")
                break

if __name__ == "__main__":
    main()