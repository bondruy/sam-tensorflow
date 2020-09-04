import tensorflow as tf
import config
shape_r_out = config.DIMS['image_out_size_salicon'][0]
shape_c_out = config.DIMS['image_out_size_salicon'][1]


def kld(y_true, y_pred, eps=1e-7):
    """This function computes the Kullback-Leibler divergence between ground
       truth saliency maps and their predictions. Values are first divided by
       their sum for each image to yield a distribution that adds to 1.

    Args:
        y_true (tensor, float32): A 4d tensor that holds the ground truth
                                  saliency maps with values between 0 and 255.
        y_pred (tensor, float32): A 4d tensor that holds the predicted saliency
                                  maps with values between 0 and 1.
        eps (scalar, float, optional): A small factor to avoid numerical
                                       instabilities. Defaults to 1e-7.

    Returns:
        tensor, float32: A 0D tensor that holds the averaged error.
    """

    sum_per_image = tf.reduce_sum(y_true, axis=(1, 2, 3), keep_dims=True)
    y_true /= eps + sum_per_image

    sum_per_image = tf.reduce_sum(y_pred, axis=(1, 2, 3), keep_dims=True)
    y_pred /= eps + sum_per_image

    loss = y_true * tf.log(eps + y_true / (eps + y_pred))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=(1, 2, 3)))

    return loss


# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred):
    sum_y_true = tf.reduce_sum(y_true, axis=[1, 2, 3], keep_dims=True)
    sum_y_pred = tf.reduce_sum(y_pred, axis=[1, 2, 3], keep_dims=True)

    y_true /= (sum_y_true + 1e-7)
    y_pred /= (sum_y_pred + 1e-7)

    N = shape_r_out * shape_c_out

    sum_prod = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    sum_x = tf.reduce_sum(y_true, axis=[1, 2, 3])
    sum_y = tf.reduce_sum(y_pred * y_pred, axis=[1, 2, 3])
    sum_x_square = tf.reduce_sum(tf.square(y_true), axis=[1, 2, 3])
    sum_y_square = tf.reduce_sum(tf.square(y_pred), axis=[1, 2, 3])

    num = sum_prod - ((sum_x * sum_y) / N)
    den = tf.sqrt((sum_x_square - tf.square(sum_x) / N) * (sum_y_square - tf.square(sum_y) / N))

    return -tf.reduce_mean(num / den)


# Normalized Scanpath Saliency Loss
def nss(y_true, y_pred):
    y_mean, y_var = tf.nn.moments(y_pred, [1, 2, 3], keep_dims=True)
    y_std = tf.sqrt(y_var)

    y_pred = (y_pred - y_mean) / (y_std + 1e-7)

    return -tf.reduce_mean((tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3]) / tf.reduce_sum(y_true, axis=[1, 2, 3])))
