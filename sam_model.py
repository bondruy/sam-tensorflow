import os

import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from ConvLSTM import AttentiveConvLSTM
from gaussian_prior import LearningPrior
from vgg_net import vgg_net
from tensorflow.python.tools import freeze_graph
from tensorflow.tools.graph_transforms import TransformGraph

import config
import download
import loss


class SAMNET:
    """The class representing the SAM-Net based on the VGG16 model. It
       implements a definition of the computational graph, as well as
       functions related to network training.
    """

    def __init__(self):
        self._output = None
        self._mapping = {}

        self._nb_gaussian = config.PARAMS["nb_gaussian"]
        self.nb_timestep = config.PARAMS["nb_timestep"]
        self.shape_r_gt = config.PARAMS["shape_r_gt"]
        self.shape_c_gt = config.PARAMS["shape_c_gt"]
        self.shape_r_out = config.DIMS["image_out_size_salicon"][0]
        self.shape_c_out = config.DIMS["image_out_size_salicon"][1]

        if config.PARAMS["device"] == "gpu":
            self._data_format = "channels_first"
            self._channel_axis = 1
            self._dims_axis = (2, 3)
        elif config.PARAMS["device"] == "cpu":
            self._data_format = "channels_last"
            self._channel_axis = 3
            self._dims_axis = (1, 2)

        self.attionconvlstm = AttentiveConvLSTM([self.shape_r_gt, self.shape_c_gt], 3, 512, 512, 512,
                                                data_format=self._data_format)
        self.priorlearing1 = LearningPrior(self._nb_gaussian, self.shape_r_gt, self.shape_c_gt, name='w1')
        self.priorlearing2 = LearningPrior(self._nb_gaussian, self.shape_r_gt, self.shape_c_gt, name='w2')

    def _encoder(self, images):
        """The encoder of the model consists of a pretrained VGG16 architecture
           with 13 convolutional layers. All dense layers are discarded and the
           last 3 layers are dilated at a rate of 2 to account for the omitted
           downsampling.

        Args:
            images (tensor, float32): A 4D tensor that holds the RGB image
                                      batches used as input to the network.
        """

        imagenet_mean = tf.constant([103.939, 116.779, 123.68])
        imagenet_mean = tf.reshape(imagenet_mean, [1, 1, 1, 3])

        images -= imagenet_mean

        if self._data_format == "channels_first":
            images = tf.transpose(images, (0, 3, 1, 2))

        features = vgg_net(images, self._data_format)
        self._output = features

    def _attenion_convlstm(self, features):
        x_tile = tf.tile(tf.layers.Flatten()(features), [1, self.nb_timestep])
        x_tile = tf.reshape(x_tile, [-1, self.nb_timestep, 512, self.shape_r_gt, self.shape_c_gt])
        inital_state = self.attionconvlstm.get_initial_states(x_tile)
        _, state = dynamic_rnn(self.attionconvlstm, x_tile, initial_state=inital_state, dtype=x_tile.dtype)
        self._output = state.h

    def _prior_learing(self, features):
        priors1 = self.priorlearing1.forword(features)
        concateneted = tf.concat([features, priors1], axis=1)
        learned_priors1 = tf.layers.conv2d(concateneted, 512, 5, padding="same", activation=tf.nn.relu,
                                           dilation_rate=4, data_format=self._data_format,
                                           name="conv/priors1")

        priors2 = self.priorlearing2.forword(learned_priors1)
        concateneted = tf.concat([learned_priors1, priors2], axis=1)
        learned_priors2 = tf.layers.conv2d(concateneted, 512, 5, padding="same", activation=tf.nn.relu,
                                           dilation_rate=4, data_format=self._data_format,
                                           name="conv/priors2")

        # Final Convolutional Layer
        outs = tf.layers.conv2d(learned_priors2, 1, 1, padding="same",
                                activation=tf.nn.relu, data_format=self._data_format,
                                name="conv/decoder")

        b_s = tf.shape(features)[0]
        outs = self._upsample(outs, [b_s, 1, self.shape_r_out, self.shape_c_out], 1)

        if self._data_format == "channels_first":
            outs = tf.transpose(outs, (0, 2, 3, 1))

        self._output = outs

    def _upsample(self, stack, shape, factor):
        """This function resizes the input to a desired shape via the
           bilinear upsampling method.

        Args:
            stack (tensor, float32): A 4D tensor with the function input.
            shape (tensor, int32): A 1D tensor with the reference shape.
            factor (scalar, int): An integer denoting the upsampling factor.

        Returns:
            tensor, float32: A 4D tensor that holds the activations after
                             bilinear upsampling of the input.
        """

        if self._data_format == "channels_first":
            stack = tf.transpose(stack, (0, 2, 3, 1))

        stack = tf.image.resize_bilinear(stack, (shape[self._dims_axis[0]] * factor,
                                                 shape[self._dims_axis[1]] * factor))

        if self._data_format == "channels_first":
            stack = tf.transpose(stack, (0, 3, 1, 2))

        return stack

    def _normalize(self, maps, eps=1e-7):
        """This function normalizes the output values to a range
           between 0 and 1 per saliency map.

        Args:
            maps (tensor, float32): A 4D tensor that holds the model output.
            eps (scalar, float, optional): A small factor to avoid numerical
                                           instabilities. Defaults to 1e-7.
        """

        min_per_image = tf.reduce_min(maps, axis=(1, 2, 3), keep_dims=True)
        maps -= min_per_image

        max_per_image = tf.reduce_max(maps, axis=(1, 2, 3), keep_dims=True)
        maps = tf.divide(maps, eps + max_per_image, name="output")

        self._output = maps

    def _pretraining(self):
        """The first 26 variables of the model here are based on the VGG16
           network. Therefore, their names are matched to the ones of the
           pretrained VGG16 checkpoint for correct initialization.
        """
        for var in tf.global_variables()[2:28]:
            key = var.name.split("/", 1)[1]
            key = key.replace("kernel:0", "weights")
            key = key.replace("bias:0", "biases")
            self._mapping[key] = var

    def forward(self, images):
        """Public method to forward RGB images through the whole network
           architecture and retrieve the resulting output.

        Args:
            images (tensor, float32): A 4D tensor that holds the values of the
                                      raw input images.

        Returns:
            tensor, float32: A 4D tensor that holds the values of the
                             predicted saliency maps.
        """

        self._encoder(images)
        self._attenion_convlstm(self._output)
        self._prior_learing(self._output)
        self._normalize(self._output)

        return self._output

    def train(self, ground_truth_map, ground_truth_fixation, predicted_maps, learning_rate):
        """Public method to define the loss function and optimization
           algorithm for training the model.

        Args:
            ground_truth (tensor, float32): A 4D tensor with the ground truth.
            predicted_maps (tensor, float32): A 4D tensor with the predictions.
            learning_rate (scalar, float): Defines the learning rate.

        Returns:
            object: The optimizer element used to train the model.
            tensor, float32: A 0D tensor that holds the averaged error.
        """

        kld = loss.kld(ground_truth_map, predicted_maps)
        cc = loss.correlation_coefficient(ground_truth_map, predicted_maps)
        nss = loss.nss(ground_truth_fixation, predicted_maps)
        error = 10 * kld + 2 * cc + nss
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        optimizer = optimizer.minimize(error)

        return optimizer, error

    def save(self, saver, sess, dataset, path, device):
        """This saves a model checkpoint to disk and creates
           the folder if it doesn't exist yet.

        Args:
            saver (object): An object for saving the model.
            sess (object): The current TF training session.
            path (str): The path used for saving the model.
            device (str): Represents either "cpu" or "gpu".
        """

        os.makedirs(path, exist_ok=True)

        saver.save(sess, path + "model_%s_%s.ckpt" % (dataset, device),
                   write_meta_graph=False, write_state=False)

    def restore(self, sess, dataset, paths, device):
        """This function allows continued training from a prior checkpoint and
           training from scratch with the pretrained VGG16 weights. In case the
           dataset is either CAT2000 or MIT1003, a prior checkpoint based on
           the SALICON dataset is required.

        Args:
            sess (object): The current TF training session.
            dataset ([type]): The dataset used for training.
            paths (dict, str): A dictionary with all path elements.
            device (str): Represents either "cpu" or "gpu".

        Returns:
            object: A saver object for saving the model.
        """

        model_name = "model_%s_%s" % (dataset, device)
        salicon_name = "model_salicon_%s" % device
        vgg16_name = "vgg16_hybrid"

        ext1 = ".ckpt.data-00000-of-00001"
        ext2 = ".ckpt.index"

        saver = tf.train.Saver()

        if os.path.isfile(paths["latest"] + model_name + ext1) and \
           os.path.isfile(paths["latest"] + model_name + ext2):
            saver.restore(sess, paths["latest"] + model_name + ".ckpt")
        elif dataset in ("mit1003", "cat2000", "dutomron",
                         "pascals", "osie", "fiwi"):
            if os.path.isfile(paths["best"] + salicon_name + ext1) and \
               os.path.isfile(paths["best"] + salicon_name + ext2):
                saver.restore(sess, paths["best"] + salicon_name + ".ckpt")
            else:
                raise FileNotFoundError("Train model on SALICON first")
        else:
            if not (os.path.isfile(paths["weights"] + vgg16_name + ext1) or
                    os.path.isfile(paths["weights"] + vgg16_name + ext2)):
                download.download_pretrained_weights(paths["weights"],
                                                     "vgg16_hybrid")
            self._pretraining()

            loader = tf.train.Saver(self._mapping)
            loader.restore(sess, paths["weights"] + vgg16_name + ".ckpt")

        return saver

    def optimize(self, sess, dataset, path, device):
        """The best performing model is frozen, optimized for inference
           by removing unneeded training operations, and written to disk.

        Args:
            sess (object): The current TF training session.
            path (str): The path used for saving the model.
            device (str): Represents either "cpu" or "gpu".

        .. seealso:: https://bit.ly/2VBBdqQ and https://bit.ly/2W7YqBa
        """

        model_name = "model_%s_%s" % (dataset, device)
        model_path = path + model_name

        tf.train.write_graph(sess.graph.as_graph_def(),
                             path, model_name + ".pbtxt")

        freeze_graph.freeze_graph(model_path + ".pbtxt", "", False,
                                  model_path + ".ckpt", "output",
                                  "save/restore_all", "save/Const:0",
                                  model_path + ".pb", True, "")

        os.remove(model_path + ".pbtxt")

        graph_def = tf.GraphDef()

        with tf.gfile.Open(model_path + ".pb", "rb") as file:
            graph_def.ParseFromString(file.read())

        transforms = ["remove_nodes(op=Identity)",
                      "merge_duplicate_nodes",
                      "strip_unused_nodes",
                      "fold_constants(ignore_errors=true)"]

        optimized_graph_def = TransformGraph(graph_def,
                                             ["input"],
                                             ["output"],
                                             transforms)

        tf.train.write_graph(optimized_graph_def,
                             logdir=path,
                             as_text=False,
                             name=model_name + ".pb")


