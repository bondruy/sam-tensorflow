import tensorflow as tf
import numpy as np


class LearningPrior:
    def __init__(self, nb_gaussian, shape_r_gt, shape_c_gt, name='w', init=tf.random_normal_initializer()):
        self._nb_gaussian = nb_gaussian
        self.height = shape_r_gt
        self.width = shape_c_gt
        # self.gaussian = np.zeros((self.b_s, self._nb_gaussian, self.height, self.width))
        e = self.height / self.width
        e1 = (1 - e) / 2
        e2 = e1 + e
        x_t = tf.matmul(tf.ones((self.height, 1)), tf.expand_dims(self._linspace(0, 1.0, self.width), 0))
        y_t = tf.matmul(tf.expand_dims(self._linspace(e1, e2, self.height), 1), tf.ones((1, self.width)))

        self.x_t = tf.tile(tf.expand_dims(x_t, dim=-1), [1, 1, self._nb_gaussian])
        self.y_t = tf.tile(tf.expand_dims(y_t, dim=-1), [1, 1, self._nb_gaussian])

        self._init = init
        self._name = name
        with tf.variable_scope("prior"):
            self.w = tf.get_variable(name, shape=[self._nb_gaussian * 4], initializer=self._init,
                                     trainable=True)

    def forword(self, features):
        b_s = tf.shape(features)[0]
        mu_x = self.w[:self._nb_gaussian]
        mu_y = self.w[self._nb_gaussian:self._nb_gaussian * 2]
        sigma_x = self.w[self._nb_gaussian * 2:self._nb_gaussian * 3]
        sigma_y = self.w[self._nb_gaussian * 3:]

        mu_x = tf.clip_by_value(mu_x, 0.25, 0.75)
        mu_y = tf.clip_by_value(mu_y, 0.35, 0.65)

        sigma_x = tf.clip_by_value(sigma_x, 0.1, 0.9)
        sigma_y = tf.clip_by_value(sigma_y, 0.2, 0.8)

        gaussian = 1 / (2 * np.pi * sigma_x * sigma_y + 1e-7) * \
                            tf.exp(-((self.x_t - mu_x) ** 2 / (2 * sigma_x ** 2 + 1e-7) +
                            (self.y_t - mu_y) ** 2 / (2 * sigma_y ** 2 + 1e-7)))

        gaussian = tf.transpose(gaussian, [2, 0, 1])
        max_gauss = tf.reduce_max(gaussian, axis=[1, 2], keep_dims=True)

        gaussian = gaussian / max_gauss

        output = tf.tile(tf.expand_dims(gaussian, dim=0), [b_s, 1, 1, 1])

        return output

    @staticmethod
    def _linspace(start, stop, num):
        # produces results identical to:
        # np.linspace(start, stop, num)
        # start = np.cast(start, np.float32)
        # stop = np.cast(stop, np.float32)
        # num = np.cast(num, np.float32)
        step = (stop - start) / (num - 1)
        _lin = np.arange(num, dtype=np.float32) * step + start
        _lin = tf.convert_to_tensor(_lin, dtype=tf.float32)
        return _lin


if __name__ == '__main__':
    p = LearningPrior(16, 30, 40)
    x = tf.placeholder(tf.float32, shape=[10, 256, 30, 40])
    w = p.forword(x)
    with tf.Session() as sess:
        input = np.ones([10, 256, 30, 40], dtype=np.float32)
        sess.run(tf.global_variables_initializer())
        x = sess.run(w, feed_dict={x: input})
        print(x)
