import tensorflow as tf


def vgg_net(images, _data_format):
    layer01 = tf.layers.conv2d(images, 64, 3,
                               padding="same",
                               activation=tf.nn.relu,
                               data_format=_data_format,
                               name="conv1/conv1_1")

    layer02 = tf.layers.conv2d(layer01, 64, 3,
                               padding="same",
                               activation=tf.nn.relu,
                               data_format=_data_format,
                               name="conv1/conv1_2")

    layer03 = tf.layers.max_pooling2d(layer02, 2, 2,
                                      data_format=_data_format)

    layer04 = tf.layers.conv2d(layer03, 128, 3,
                               padding="same",
                               activation=tf.nn.relu,
                               data_format=_data_format,
                               name="conv2/conv2_1")

    layer05 = tf.layers.conv2d(layer04, 128, 3,
                               padding="same",
                               activation=tf.nn.relu,
                               data_format=_data_format,
                               name="conv2/conv2_2")

    layer06 = tf.layers.max_pooling2d(layer05, 2, 2,
                                      data_format=_data_format)

    layer07 = tf.layers.conv2d(layer06, 256, 3,
                               padding="same",
                               activation=tf.nn.relu,
                               data_format=_data_format,
                               name="conv3/conv3_1")

    layer08 = tf.layers.conv2d(layer07, 256, 3,
                               padding="same",
                               activation=tf.nn.relu,
                               data_format=_data_format,
                               name="conv3/conv3_2")

    layer09 = tf.layers.conv2d(layer08, 256, 3,
                               padding="same",
                               activation=tf.nn.relu,
                               data_format=_data_format,
                               name="conv3/conv3_3")

    layer10 = tf.layers.max_pooling2d(layer09, 2, 2,
                                      data_format=_data_format)

    layer11 = tf.layers.conv2d(layer10, 512, 3,
                               padding="same",
                               activation=tf.nn.relu,
                               data_format=_data_format,
                               name="conv4/conv4_1")

    layer12 = tf.layers.conv2d(layer11, 512, 3,
                               padding="same",
                               activation=tf.nn.relu,
                               data_format=_data_format,
                               name="conv4/conv4_2")

    layer13 = tf.layers.conv2d(layer12, 512, 3,
                               padding="same",
                               activation=tf.nn.relu,
                               data_format=_data_format,
                               name="conv4/conv4_3")

    layer14 = tf.layers.max_pooling2d(layer13, 2, 1,
                                      padding="same",
                                      data_format=_data_format)

    layer15 = tf.layers.conv2d(layer14, 512, 3,
                               padding="same",
                               activation=tf.nn.relu,
                               dilation_rate=2,
                               data_format=_data_format,
                               name="conv5/conv5_1")

    layer16 = tf.layers.conv2d(layer15, 512, 3,
                               padding="same",
                               activation=tf.nn.relu,
                               dilation_rate=2,
                               data_format=_data_format,
                               name="conv5/conv5_2")

    layer17 = tf.layers.conv2d(layer16, 512, 3,
                               padding="same",
                               activation=tf.nn.relu,
                               dilation_rate=2,
                               data_format=_data_format,
                               name="conv5/conv5_3")

    return layer17
