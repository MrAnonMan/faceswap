#!/usr/bin/env python3
""" DeepFaceLab SAEHD Model
    Based on https://github.com/iperov/DeepFaceLab
"""
from math import ceil

import numpy as np

import keras

from keras import backend as K
from keras.layers import Conv2DTranspose, Dense, Flatten, Input, LeakyReLU, Reshape, UpSampling2D

from lib.model.layers import DenseNorm
from lib.model.nn_blocks import (Concatenate, Conv2D, Conv2DBlock, Conv2DOutput, UpscaleBlock,
                                 ResidualBlock)
from ._base import ModelBase, KerasModel, losses, logger, k_losses


# TODO Add note to use RMS Prop for stock settings
# TODO TrueFacePower is only used in training, not inference. Check this
# TODO Conv2D Transpose to NNBlocks
# TODO Something weird in original implementation around res doubler.
# resolution moved from res // 16 * 16 to res // 32 * 32
# TODO Split resolution DSSIM
# TODO LIAE Inters look wrong
# TODO Res changes on dssim

class Model(ModelBase):
    """ SAE Model from DFL """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_size = self._round_up(self.config["input_size"], factor=16)
        self.input_shape = (input_size, input_size, 3)
        self.trainer = "dfl_sae"

        self._architecture = self.config["architecture"].lower()
        self._ae_dims = self._get_ae_dims()
        self._encoder_dims = self._round_up(self.config["encoder_dims"])
        self._decoder_dims = self._round_up(self.config["decoder_dims"])
        self._use_mask = self.config.get("learn_mask", False)

        self._inputs = dict()
        self._outputs = dict()
        self._full_model = None

    @property
    def _true_face_power(self):
        """ float: The true face power config item reduced to 0.0 - 1.0 range. """
        return self.config["true_face_power"] / 100.0

    @property
    def _gan_power(self):
        """ float: The GAN power config item reduced to 0.0 - 1.0 range. """
        return self.config["gan_power"] / 100.0

    @classmethod
    def _round_up(cls, value, factor=2):
        """ Round the given integer to the nearest given factor.

        Parameters
        ----------
        value: int
            The integer to be scaled to a power of 2
        factor: int, optional
            The factor to scale to. Default: 2

        Returns
        -------
        int
            The input value rounded up to the nearest factor
        """
        return int(ceil(value / factor) * factor)

    def _get_ae_dims(self):
        retval = self.config["autoencoder_dims"]
        if retval == 0:
            retval = 256 if self._architecture == "liae" else 512
        return retval

    def build_model(self, inputs):
        """ Build the DFL-SAEHD Model """
        self._inputs["face"] = inputs
        outputs = getattr(self, "_build_{}".format(self._architecture))(inputs)
        output_shape = K.int_shape(outputs[0])[1:]

        self._inputs["gan"] = Input(shape=output_shape, name="gan_discrim")
        inputs.append(self._inputs["gan"])

        gan = UNetPatchDiscriminator(output_shape).build()
        gan_gen = gan(outputs[0])
        gan_dis = gan(self._inputs["gan"])

        self._outputs["gan"] = [gan_gen, gan_dis]
        outputs.extend(self._outputs["gan"])

        autoencoder = KerasModel(inputs,
                                 outputs,
                                 name="{}_{}".format(self.name, self._architecture))
        return autoencoder

    def _build_df(self, inputs):
        encoder = self.encoder()
        inter = self.inter(encoder.output_shape[1:])

        inter_out_shape = inter.output_shape[1:]
        decoder_a = self.decoder(inter_out_shape, side="a")
        decoder_b = self.decoder(inter_out_shape, side="b")

        encoder_a = encoder(inputs[0])
        encoder_b = encoder(inputs[1])
        inter_a = inter(encoder_a)
        inter_b = inter(encoder_b)
        self._outputs["face"] = [decoder_a(inter_a), decoder_b(inter_b)]
        self._outputs["swapped"] = decoder_b(inter_a)
        self._outputs["true_face"] = self.discriminator_df(inter_out_shape)

        outputs = [self._outputs["face"][0],
                   self._outputs["face"][1],
                   self._outputs["swapped"],
                   self._outputs["true_face"](inter_a),
                   self._outputs["true_face"](inter_b)]
        return outputs

    def _build_liae(self, inputs):
        encoder = self.encoder()

        inter_ab = self.inter(encoder.output_shape[1:], side="ab")
        inter_b = self.inter(encoder.output_shape[1:], side="b")

        # TODO
        # inter_out_shapes = (np.array(inter_ab.output_shape[1:]) + np.array(inter_b.output_shape[1:])).tolist()
        inter_out_shape = (np.array(inter_ab.output_shape[1:]) * (1, 1, 2)).tolist()
        decoder = self.decoder(inter_out_shape)

        encoder_a = encoder(inputs[0])
        encoder_b = encoder(inputs[1])

        inter_a = Concatenate()([inter_ab(encoder_a), inter_ab(encoder_a)])
        inter_b = Concatenate()([inter_b(encoder_b), inter_ab(encoder_b)])

        inter_swap = Concatenate()[inter_ab(encoder_b), inter_ab(encoder_b)]
        swap_model = decoder(inter_swap)(encoder_a)

        outputs = [decoder(inter_a), decoder(inter_b), swap_model]
        return outputs

    def _configure_options(self):
        """ Override to add additional losses as requested.

        Configure the options for the Optimizer and Loss Functions.

        Returns the request optimizer, and sets the loss parameters in :attr:`_loss`.

        Returns
        :class:`keras.optimizers.Optimizer`
            The request optimizer
        """
        self._full_model = self._model
        self._set_training_model()
        optimizer = super()._configure_options()
        self._loss.__init__()
        # TODO Calc filter
        base_loss_model = KerasModel(self._model.inputs[:2],
                                     self._model.outputs[:2],
                                     name="base_model")
        self._loss.configure(base_loss_model)
        self._set_additional_loss()
        # TODO Mask channel

        return optimizer

    def _set_training_model(self):
        """ Collate a version of the full model only containing those required inputs and outputs
        for the selected user options """
        inputs = [inp for inp, name in zip(self._model.input, self._model.input_names)
                  if name != "gan_discrim" or (name == "gan_discrim" and self._gan_power > 0.0)]
        logger.debug("Training inputs: %s", inputs)

        outputs = [lyr for lyr, name in zip(self._model.output, self._model.output_names)
                   if name.startswith("decoder")
                   or (name.startswith("discriminator_df") and self._true_face_power > 0.0)
                   or (name.startswith("patch_discriminator") and self._gan_power > 0.0)]
        if self.config["face_style_power"] == 0.0 and self.config["bg_style_power"] == 0.0:
            # TODO Remove the last decoder output as this is the swap model output. Need to
            # check counts with learn mask attribute
            outputs.pop(2)

        logger.debug("Training outputs: %s", outputs)
        self._model = KerasModel(inputs, outputs, name="{}_training".format(self.name))

    def _set_additional_loss(self):
        """ Set the additional loss functions for extra outputs """
        # TODO
        # print(self._loss.functions)
        # print(self._model.output_names)
        if self.config["face_style_power"] > 0.0 or self.config["bg_style_power"] > 0.0:
            swap_loss = losses.LossWrapper()
            mask_channel = 3
            if self.config["face_style_power"] > 0.0:
                swap_loss.add_loss(StyleLoss(loss_weight=self.config["face_style_power"] * 100.0),
                                   mask_channel=mask_channel)
                mask_channel += 1
            if self.config["bg_style_power"] > 0.0:
                swap_loss.add_loss(self._loss.loss_function,
                                   weight=self.config["bg_style_power"],
                                   mask_channel=mask_channel)
                self._loss._add_l2_regularization_term(swap_loss, 3)
            self._loss.add_function_to_output("decoder_df_b_1", swap_loss)

        if self._architecture == "df" and self._true_face_power > 0.0:
            # TODO Mask shouldn't input here.
            self._loss.add_function_to_output("discriminator_df", k_losses.binary_crossentropy)

        if self._gan_power > 0.0:
            # TODO Mask shouldn't input here
            # GAN Takes 1 version with output of A>A and 1 with source tgt image
            # TODO The loss function
            self._loss.add_function_to_output("patch_discriminator", k_losses.mean_squared_error)

    def encoder(self):
        """ DFL SAEHD Encoder Network"""
        input_ = Input(shape=self.input_shape)
        var_x = input_

        for idx in range(4):
            filters = self._encoder_dims * min(2**idx, 8)
            var_x = Conv2DBlock(filters)(var_x)
        var_x = Flatten()(var_x)
        return KerasModel(input_, var_x, name="encoder_{}".format(self._architecture))

    def inter(self, input_shape, side=None):
        """ DFL SAEHD Intermediate Network """
        input_ = Input(shape=input_shape)
        lowest_dense_res = self.input_shape[0] // (32 if self.config["res_double"] else 16)
        ae_out_channels = self._ae_dims if self._architecture == "df" else self._ae_dims * 2

        var_x = input_
        if self.config["dense_norm"]:
            var_x = DenseNorm()(var_x)
        var_x = Dense(self._ae_dims)(var_x)
        var_x = Dense(lowest_dense_res * lowest_dense_res * ae_out_channels)(var_x)
        var_x = Reshape((lowest_dense_res, lowest_dense_res, ae_out_channels))(var_x)
        var_x = UpscaleBlock(ae_out_channels)(var_x)

        name = "inter_{}".format(self._architecture)
        name = "{}{}".format(name, "_{}".format(side) if side is not None else "")
        return KerasModel(input_, var_x, name=name)

    def decoder(self, input_shape, side=None):
        """ DFL SAEHD Decoder Network """
        input_ = Input(shape=input_shape)

        var_x = UpscaleBlock(self._decoder_dims * 8, res_block_follows=True)(input_)
        var_x = ResidualBlock(self._decoder_dims * 8)(var_x)
        var_x = UpscaleBlock(self._decoder_dims * 4, res_block_follows=True)(var_x)
        var_x = ResidualBlock(self._decoder_dims * 4)(var_x)
        var_x = UpscaleBlock(self._decoder_dims * 2, res_block_follows=True)(var_x)
        var_x = ResidualBlock(self._decoder_dims * 2)(var_x)

        if self.config["res_double"]:
            var_x0 = Conv2DOutput(3, 1)(var_x)
            var_x0 = UpSampling2D()(var_x0)
            var_x1 = Conv2DOutput(3, 3)(var_x)
            var_x1 = UpSampling2D()(var_x1)
            var_x2 = Conv2DOutput(3, 3)(var_x)
            var_x2 = UpSampling2D()(var_x2)
            var_x3 = Conv2DOutput(3, 3)(var_x)
            var_x3 = UpSampling2D()(var_x3)

            tile_shape = (1, self.input_shape[0] // 2, self.input_shape[0] // 2, 1)
            var_z0 = K.concatenate([K.concatenate([K.ones((1, 1, 1, 1)),
                                                   K.zeros((1, 1, 1, 1))], axis=2),
                                    K.concatenate([K.zeros((1, 1, 1, 1)),
                                                   K.zeros((1, 1, 1, 1))], axis=2)], axis=1)
            var_z0 = K.tile(var_z0, tile_shape)
            var_z1 = K.concatenate([K.concatenate([K.zeros((1, 1, 1, 1)),
                                                   K.ones((1, 1, 1, 1))], axis=2),
                                    K.concatenate([K.zeros((1, 1, 1, 1)),
                                                   K.zeros((1, 1, 1, 1))], axis=2)], axis=1)
            var_z1 = K.tile(var_z1, tile_shape)
            var_z2 = K.concatenate([K.concatenate([K.zeros((1, 1, 1, 1)),
                                                   K.zeros((1, 1, 1, 1))], axis=2),
                                    K.concatenate([K.ones((1, 1, 1, 1)),
                                                   K.zeros((1, 1, 1, 1))], axis=2)], axis=1)
            var_z2 = K.tile(var_z2, tile_shape)
            var_z3 = K.concatenate([K.concatenate([K.zeros((1, 1, 1, 1)),
                                                   K.zeros((1, 1, 1, 1))], axis=2),
                                    K.concatenate([K.zeros((1, 1, 1, 1)),
                                                   K.ones((1, 1, 1, 1))], axis=2)], axis=1)
            var_z3 = K.tile(var_z3, tile_shape)
            var_x = (var_x0 * var_z0) + (var_x1 * var_z1) + (var_x2 * var_z2) + (var_x3 * var_z3)
        else:
            var_x = Conv2DOutput(3, 1)(var_x)

        name = "decoder_{}".format(self._architecture)
        name = "{}{}".format(name, "_{}".format(side) if side is not None else "")
        return KerasModel(input_, var_x, name=name)

    def discriminator_df(self, input_shape):
        """ True Face Power Code Discriminator for DF Architecture. """
        input_ = Input(shape=input_shape)
        code_res = (self.input_shape[0] // (16 if self.config["res_double"] else 8))

        var_x = input_
        for idx in range(1 + code_res // 8):
            filters = 256 * min(2**idx, 8)
            kernel_size = 4 if idx == 0 else 3
            var_x = Conv2DBlock(filters, kernel_size=kernel_size)(var_x)
        var_x = Conv2D(1, 1, padding="VALID")(var_x)
        return KerasModel(input_, var_x, name="discriminator_df")


class UNetPatchDiscriminator():  # pylint:disable=too-few-public-methods
    def __init__(self, input_shape):
        self._input_shape = input_shape
        self._layers = self._get_layers()

    def _get_layers(self, max_layers=6):
        """ Obtain the best configuration of layers using only 3x3 convolutions for the target
        patch size

        Parameters
        ----------
        max_layers: int, optional
            The maximum number of layers to return. Default: `6`
        """
        target_patch_size = self._input_shape[0] // 16
        samples = {}
        for layers_count in range(1, max_layers + 1):
            val = 1 << (layers_count - 1)
            while val != 0:
                val -= 1
                layers = []
                sum_strides = 0
                for idx in range(layers_count - 1):
                    strides = 1 + (1 if val & (1 << idx) != 0 else 0)
                    layers.append([3, strides])
                    sum_strides += strides
                layers.append([3, 2])
                sum_strides += 2
                receptive_field_size = self._calc_receptive_field_size(layers)
                s_rf = samples.get(receptive_field_size, None)

                if s_rf is None:
                    samples[receptive_field_size] = (layers_count, sum_strides, layers)
                else:
                    if layers_count < s_rf[0] or (layers_count == s_rf[0]
                                                  and sum_strides > s_rf[1]):
                        samples[receptive_field_size] = (layers_count, sum_strides, layers)

        keys = sorted(list(samples.keys()))
        result = keys[np.abs(np.array(keys) - target_patch_size).argmin()]
        return samples[result][2]

    @classmethod
    def _calc_receptive_field_size(cls, layers):
        """
        result the same as https://fomoro.com/research/article/receptive-field-calculatorindex.html
        """
        receptive_fields = 0
        target_size = 1
        for idx, (k, size) in enumerate(layers):
            if idx == 0:
                receptive_fields = k
            else:
                receptive_fields += (k - 1) * target_size
            target_size *= size
        return receptive_fields

    def build(self):
        """ Build the discriminator Model.

        Returns
        -------
        :class:`keras.models.Model`
            The built discriminator model
        """

        input_ = Input(shape=self._input_shape)
        base_filters = 16

        var_x = Conv2DBlock(base_filters, kernel_size=1, strides=1, padding="valid")(input_)
        encodings = []
        for idx, (kernel_size, strides) in enumerate(self._layers):
            encodings.append(var_x)
            filters = min(base_filters * 2**(idx + 1), 512)
            var_x = Conv2DBlock(filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                res_block_follows=True)(var_x)
            var_x = ResidualBlock(filters)(var_x)

        center_out = Conv2DBlock(1, kernel_size=1, strides=1, padding="valid")(var_x)
        var_x = Conv2DBlock(min(base_filters * 2**len(self._layers), 512),
                            kernel_size=1,
                            strides=1,
                            padding='VALID')(var_x)

        for idx, (enc, (kernel_size, strides)) in reversed(list(enumerate(zip(encodings,
                                                                              self._layers)))):
            filters = min(base_filters * 2**(idx + 1), 512)
            transpose_filters = min(base_filters * 2**idx, 512)
            var_x = Conv2DTranspose(transpose_filters,
                                    kernel_size,
                                    strides=strides,
                                    padding="same")(var_x)
            var_x = LeakyReLU(0.1)(var_x)
            var_x = Concatenate()([enc, var_x])
            var_x = ResidualBlock(filters)(var_x)

        var_x = Conv2DBlock(1, kernel_size=1, padding="valid")(var_x)
        return KerasModel(input_, [center_out, var_x], name="patch_discriminator")


class StyleLoss(keras.losses.Loss):  # pylint:disable=too-few-public-methods
    """ Style Loss Function.

    Parameters
    ----------
    loss_weight: float, optional
        The weight to apply to the loss. Default: `1.0`
    """
    def __init__(self, loss_weight=1.0):
        super().__init__(name="StyleLoss")
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        """ Call the Style Loss Loss Function.

        Parameters
        ----------
        y_true: tensor or variable
            The ground truth value
        y_pred: tensor or variable
            The predicted value

        Returns
        -------
        tensor
            The Style Loss value
        """
        true_mean = K.mean(y_true, axis=[1, 2], keepdims=True)
        true_var = K.var(y_true, axis=[1, 2], keepdims=True)
        true_std = K.sqrt(true_var + 1e-5)

        pred_mean = K.mean(y_pred, axis=[1, 2], keepdims=True)
        pred_var = K.var(y_pred, axis=[1, 2], keepdims=True)
        pred_std = K.sqrt(pred_var + 1e-5)

        mean_loss = K.sum(K.square(true_mean - pred_mean), axis=[1, 2, 3])
        std_loss = K.sum(K.square(true_std - pred_std), axis=[1, 2, 3])

        retval = (mean_loss + std_loss) * (self.loss_weight / K.int_shape(y_true)[-1])
        return retval
