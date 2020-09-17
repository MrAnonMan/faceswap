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

        dfl_model = _ModelDF if self.config["architecture"].lower() == "df" else _ModelLIAE
        self._dfl_model = dfl_model(self.input_shape,
                                    self.config,
                                    self._round_up(self.config["encoder_dims"]),
                                    self._round_up(self.config["decoder_dims"]),
                                    self.name)

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

    def save(self):
        """ Save the model to disk.

        Overrides the default save function to apply the current training weights to the full
        model rather than the training model.
        """
        training_model = self._model
        self._model = self._dfl_model.models["full"]
        self._io._save()  # pylint:disable=protected-access
        self._model = training_model

    def build_model(self, inputs):
        """ Build the DFL-SAEHD Model """
        self._dfl_model.build_full(inputs)
        return self._dfl_model.models["full"]

    def _configure_options(self):
        """ Override to add additional losses as requested.

        Configure the options for the Optimizer and Loss Functions.

        Returns the request optimizer, and sets the loss parameters in :attr:`_loss`.

        Returns
        :class:`keras.optimizers.Optimizer`
            The request optimizer
        """
        if self._dfl_model.models["full"] is None:
            self._dfl_model.models["full"] = self._model
        self._dfl_model.build_training_model()
        self._model = self._dfl_model.models["training"]

        optimizer = super()._configure_options()
        self._set_additional_loss()

        return optimizer

    def _set_additional_loss(self):
        """ Set the additional loss functions for extra outputs.

        Adds style power loses, True Face (df architecture only) and GAN
        loss to their appropriate outputs
        """
        # TODO Check the impact "learn_mask" has on all of this
        output_idx = 4 if self.config["learn_mask"] else 2
        if self.config["face_style_power"] > 0.0 or self.config["bg_style_power"] > 0.0:
            self._add_style_loss(output_idx)
            output_idx += 1

        if (self.config["architecture"].lower() == "df" and
                self._dfl_model.multipliers["true_face"] > 0.0):
            output_idx += self._add_true_face_loss(output_idx)

        if self._dfl_model.multipliers["gan"] > 0.0:
            # GAN Takes 1 version with output of A>A and 1 with source tgt image
            # TODO The loss function
            outputs = [name for name in self.model.output_names
                       if name.startswith("patch_discriminator")]
            for output in outputs:
                self._loss.add_function_to_output(output, k_losses.mean_squared_error)
            output_idx += len(outputs)

    def _add_style_loss(self, output_index):
        """ Add style loss for Face Style Power and/or Background Style Power on the given output's
        index.

        Parameters
        ----------
        output_index: int
            The index of the output layer to apply the loss to
        """
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
        self._loss.add_function_to_output(self.model.output_names[output_index], swap_loss)

    def _add_true_face_loss(self, output_index):
        """ Add True Face loss on the given output's index.

        Parameters
        ----------
        output_index: int
            The index of the output layer to apply the loss to
        """        
        outputs = [name for name in self.model.output_names
                    if name.startswith("discriminator_df")]
        for output in outputs:
            self._loss.add_function_to_output(output, k_losses.binary_crossentropy)
        output_index += len(outputs)
        return output_index

class _DFLSAEHD():
    def __init__(self, input_shape, config, encoder_dims, decoder_dims, name):
        self._input_shape = input_shape

        self._architecture = config["architecture"].lower()
        self._res_double = config["res_double"]
        self._dense_norm = config["dense_norm"]
        self._use_mask = config["learn_mask"]
        self._name = name

        self.multipliers = dict(gan=config["gan_power"] / 100.0,
                                true_face=config["true_face_power"] / 100.0,
                                face_style=config["face_style_power"],
                                bg_style=config["bg_style_power"])
        self._dims = dict(encoder=encoder_dims,
                          decoder=decoder_dims,
                          ae=config["autoencoder_dims"])
        self.models = dict(outputs=dict(), full=None, training=None, trueface=None)

    def build_full(self, inputs):
        """ Build the full DFL-SAEHD model with the selected configuration options.

        This build all sub-models in the model, regardless if they have been selected for use
        and returns the full model.

        Parameters
        ----------
        inputs: list
            A list of :class:`keras.layers.Input` tensors. This will be a list of 2 tensors (one
            for each side) each of shapes :attr:`input_shape`.

        Returns
        -------
        :class:`keras.models.Model`
            The fully built DFL-SAEHD model with the selcted configuration options
        """
        outputs = self._build_(inputs)
        output_shape = K.int_shape(outputs[0])[1:]

        gan_input = Input(shape=output_shape, name="gan_discrim")
        inputs.append(gan_input)

        gan = UNetPatchDiscriminator(output_shape).build()
        gan_gen = gan(outputs[0])
        gan_dis = gan(gan_input)
        outputs.extend([gan_gen, gan_dis])

        autoencoder = KerasModel(inputs,
                                 outputs,
                                 name="{}_{}".format(self._name, self._architecture))
        self.models["full"] = autoencoder

    def _build_(self, inputs):
        """ Build the full DFL-SAEHD Model regardless of configured options

        Override to bu all sub-models in the model, regardless if they have been selected for use.

        Parameters
        ----------
        inputs: list
            A list of :class:`keras.layers.Input` tensors. This will be a list of 2 tensors (one
            for each side) each of shapes :attr:`input_shape`.

        Returns
        -------
        list
            List of tensors as outputs to the model
        """
        raise NotImplementedError

    def build_training_model(self):
        """ Build the training version of the model based on current configuration and add to
        :attr:`_training_model`.

        This is the stock A>B Model, with Style Loss added, if requested
        """
        inputs = self.models["full"].input[:4 if self._use_mask else 2]
        logger.debug("Training inputs: %s", inputs)
        # TODO Multiscale outputs
        
        outputs = [lyr for lyr, name in zip(self.models["full"].output,
                                            self.models["full"].output_names)
                   if name.startswith("decoder")]
        if self.multipliers["face_style"] == 0.0 and self.multipliers["bg_style"] == 0.0:
            # TODO Remove the last decoder output as this is the swap model output. Need to
            # check counts with learn mask attribute
            outputs.pop(2)
        logger.debug("Training outputs: %s", outputs)
        self.models["training"] = KerasModel(inputs,
                                             outputs,
                                             name="{}_training".format(self._name))
    
        if self._architecture == "df" and self.multipliers["true_face"] > 0.0:
            true_face inputs = 

        return outputs

    def encoder(self):
        """ DFL SAEHD Encoder Network"""
        input_ = Input(shape=self._input_shape)
        var_x = input_

        for idx in range(4):
            filters = self._dims["encoder"] * min(2**idx, 8)
            var_x = Conv2DBlock(filters)(var_x)
        var_x = Flatten()(var_x)
        return KerasModel(input_, var_x, name="encoder_{}".format(self._architecture))

    def inter(self, input_shape, side=None):
        """ DFL SAEHD Intermediate Network """
        input_ = Input(shape=input_shape)
        lowest_dense_res = self._input_shape[0] // (32 if self._res_double else 16)
        ae_out_channels = self._dims["ae"] if self._architecture == "df" else self._dims["ae"] * 2

        var_x = input_
        if self._dense_norm:
            var_x = DenseNorm()(var_x)
        var_x = Dense(self._dims["ae"])(var_x)
        var_x = Dense(lowest_dense_res * lowest_dense_res * ae_out_channels)(var_x)
        var_x = Reshape((lowest_dense_res, lowest_dense_res, ae_out_channels))(var_x)
        var_x = UpscaleBlock(ae_out_channels)(var_x)

        name = "inter_{}".format(self._architecture)
        name = "{}{}".format(name, "_{}".format(side) if side is not None else "")
        return KerasModel(input_, var_x, name=name)

    def decoder(self, input_shape, side=None):
        """ DFL SAEHD Decoder Network """
        input_ = Input(shape=input_shape)

        var_x = UpscaleBlock(self._dims["decoder"] * 8, res_block_follows=True)(input_)
        var_x = ResidualBlock(self._dims["decoder"] * 8)(var_x)
        var_x = UpscaleBlock(self._dims["decoder"] * 4, res_block_follows=True)(var_x)
        var_x = ResidualBlock(self._dims["decoder"] * 4)(var_x)
        var_x = UpscaleBlock(self._dims["decoder"] * 2, res_block_follows=True)(var_x)
        var_x = ResidualBlock(self._dims["decoder"] * 2)(var_x)

        if self._res_double:
            var_x0 = Conv2DOutput(3, 1)(var_x)
            var_x0 = UpSampling2D()(var_x0)
            var_x1 = Conv2DOutput(3, 3)(var_x)
            var_x1 = UpSampling2D()(var_x1)
            var_x2 = Conv2DOutput(3, 3)(var_x)
            var_x2 = UpSampling2D()(var_x2)
            var_x3 = Conv2DOutput(3, 3)(var_x)
            var_x3 = UpSampling2D()(var_x3)

            tile_shape = (1, self._input_shape[0] // 2, self._input_shape[0] // 2, 1)
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


class _ModelDF(_DFLSAEHD):
    def _build_(self, inputs):
        """ Build the full DFL-SAEHD DF Model regardless of configured options

        This build all sub-models in the model, regardless if they have been selected for use
        and returns the full model.

        Parameters
        ----------
        inputs: list
            A list of :class:`keras.layers.Input` tensors. This will be a list of 2 tensors (one
            for each side) each of shapes :attr:`input_shape`.

        Returns
        -------
        list
            List of tensors as outputs to the model
        """
        encoder = self.encoder()
        inter = self.inter(encoder.output_shape[1:])

        inter_out_shape = inter.output_shape[1:]
        decoder_a = self.decoder(inter_out_shape, side="a")
        decoder_b = self.decoder(inter_out_shape, side="b")

        encoder_a = encoder(inputs[0])
        encoder_b = encoder(inputs[1])
        inter_a = inter(encoder_a)
        inter_b = inter(encoder_b)

        true_face = self.discriminator_df(inter_out_shape)

        outputs = [decoder_a(inter_a),  # A-A
                   decoder_b(inter_b),  # B-B
                   decoder_b(inter_a),  # A-B  (style loss)
                   true_face(inter_a),  # TrueFace(A)
                   true_face(inter_b)]  # TrueFace(B)
        return outputs

    def discriminator_df(self, input_shape):
        """ True Face Power Code Discriminator for DF Architecture. """
        input_ = Input(shape=input_shape)
        code_res = (self._input_shape[0] // (16 if self._res_double else 8))

        var_x = input_
        for idx in range(1 + code_res // 8):
            filters = 256 * min(2**idx, 8)
            kernel_size = 4 if idx == 0 else 3
            var_x = Conv2DBlock(filters, kernel_size=kernel_size)(var_x)
        var_x = Conv2D(1, 1, padding="VALID")(var_x)
        return KerasModel(input_, var_x, name="discriminator_df")


class _ModelLIAE(_DFLSAEHD):
    def _build_(self, inputs):
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


class UNetPatchDiscriminator():  # pylint:disable=too-few-public-methods
    """ GAN Discriminator for DFL SAEHD """
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
