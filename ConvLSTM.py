import tensorflow as tf
import numpy as np


class AttentiveConvLSTM(tf.nn.rnn_cell.RNNCell):
  """
     A LSTM cell with convolutions and attention.
  """

  def __init__(self, shape, kernel, nb_filters_in, nb_filters_out, nb_filters_att,
               init=tf.truncated_normal_initializer(stddev=0.05),
               inner_init=tf.orthogonal_initializer(), attentive_init=tf.zeros_initializer(),
               inner_activation=tf.sigmoid, activation=tf.tanh, data_format='channels_first', reuse=None):
    super(AttentiveConvLSTM, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._nb_filters_in = nb_filters_in
    self._nb_filters_out = nb_filters_out
    self._nb_filters_att = nb_filters_att
    self._init = init
    self._inner_init = inner_init
    self._attentive_init = attentive_init
    self._inner_activation = inner_activation
    self._activation = activation
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._nb_filters_out])
        self._feature_axis = self._size.ndims
        self._data_format = 'channels_last'
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._nb_filters_out] + shape)
        self._feature_axis = 0
        self._data_format = 'channels_first'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, states):
    x_shape = tf.shape(x)
    h_tm1 = states.h
    c_tm1 = states.c
    w_a = tf.layers.conv2d(h_tm1, self._nb_filters_att, self._kernel, padding="same",
                           activation=None,
                           kernel_initializer=self._init,
                           data_format=self._data_format,
                           name="conv/W_a")
    u_a = tf.layers.conv2d(x, self._nb_filters_att, self._kernel, padding="same",
                           activation=None,
                           kernel_initializer=self._init,
                           data_format=self._data_format,
                           name="conv/U_a")
    active_wu = tf.tanh(w_a + u_a)
    e = tf.layers.conv2d(active_wu, 1, self._kernel, padding="same",
                         activation=None,
                         use_bias=False,
                         kernel_initializer=self._attentive_init,
                         data_format=self._data_format,
                         name="conv/V_a")

    a = tf.reshape(tf.nn.softmax(tf.layers.Flatten()(e)), [x_shape[0], 1, x_shape[2], x_shape[3]])
    x_tilde = x * tf.tile(a, [1, x_shape[1], 1, 1])

    x_i = tf.layers.conv2d(x_tilde, self._nb_filters_out, self._kernel, padding="same",
                           activation=None,
                           kernel_initializer=self._init,
                           data_format=self._data_format,
                           name="conv/W_i")
    x_f = tf.layers.conv2d(x_tilde, self._nb_filters_out, self._kernel, padding="same",
                           activation=None,
                           kernel_initializer=self._init,
                           data_format=self._data_format,
                           name="conv/W_f")
    x_c = tf.layers.conv2d(x_tilde, self._nb_filters_out, self._kernel, padding="same",
                           activation=None,
                           kernel_initializer=self._init,
                           data_format=self._data_format,
                           name="conv/W_c")
    x_o = tf.layers.conv2d(x_tilde, self._nb_filters_out, self._kernel, padding="same",
                           activation=None,
                           kernel_initializer=self._init,
                           data_format=self._data_format,
                           name="conv/W_o")

    u_i = tf.layers.conv2d(h_tm1, self._nb_filters_out, self._kernel, padding="same",
                           activation=None,
                           kernel_initializer=self._inner_init,
                           data_format=self._data_format,
                           name="conv/U_i")
    u_f = tf.layers.conv2d(h_tm1, self._nb_filters_out, self._kernel, padding="same",
                           activation=None,
                           kernel_initializer=self._inner_init,
                           data_format=self._data_format,
                           name="conv/U_f")
    u_c = tf.layers.conv2d(h_tm1, self._nb_filters_out, self._kernel, padding="same",
                           activation=None,
                           kernel_initializer=self._inner_init,
                           data_format=self._data_format,
                           name="conv/U_c")
    u_o = tf.layers.conv2d(h_tm1, self._nb_filters_out, self._kernel, padding="same",
                           activation=None,
                           kernel_initializer=self._inner_init,
                           data_format=self._data_format,
                           name="conv/U_o")

    i = self._inner_activation(x_i + u_i)
    f = self._inner_activation(x_f + u_f)
    c = f * c_tm1 + i * self._activation(x_c + u_c)
    o = self._inner_activation(x_o + u_o)

    h = o * self._activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    return h, state

  def get_initial_states(self, x):
    if self._data_format is 'channels_first':
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    initial_state = tf.reduce_sum(x, axis=1)
    initial_state = tf.nn.conv2d(initial_state, tf.zeros((1, 1, self._nb_filters_out, self._nb_filters_in)),
                                 [1, 1, 1, 1], padding='SAME', data_format=data_format)
    # initial_states = [initial_state for _ in range(2)]
    initial_states = tf.nn.rnn_cell.LSTMStateTuple(initial_state, initial_state)
    return initial_states


if __name__ == '__main__':

    from tensorflow.python.ops.rnn import dynamic_rnn
    covlstm = AttentiveConvLSTM([30, 40], 3, 512, 512, 512)
    inputs = tf.placeholder(tf.float32, [10, 4] + [512] + [30, 40])
    inital_state = covlstm.get_initial_states(inputs)
    outputs, state = dynamic_rnn(covlstm, inputs, initial_state=inital_state, dtype=inputs.dtype)

    with tf.Session() as sess:
      x = np.ones([10, 4, 512, 30, 40])
      sess.run(tf.global_variables_initializer())
      outputs, state = sess.run([outputs, state], feed_dict={inputs: x})
