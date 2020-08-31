#!/usr/bin/env python3
""" Trainer for the dfl_sae model. """

# pylint:disable=too-many-lines
import numpy as np

import keras.backend as K

from lib.training_data import TrainingDataGenerator
from lib.utils import get_backend

from ._base import _Feeder, TrainerBase, logger


class Trainer(TrainerBase):
    """ Handles the feeding of training images to Faceswap models, the generation of Tensorboard
    logs and the creation of sample/time-lapse preview images.

    All Trainer plugins must inherit from this class.

    Parameters
    ----------
    model: plugin from :mod:`plugins.train.model`
        The model that will be running this trainer
    images: dict
        The file paths for the images to be trained on for each side. The dictionary should contain
        2 keys ("a" and "b") with the values being a list of full paths corresponding to each side.
    batch_size: int
        The requested batch size for iteration to be trained through the model.
    configfile: str
        The path to a custom configuration file. If ``None`` is passed then configuration is loaded
        from the default :file:`.config.train.ini` file.
    """

    def __init__(self, model, images, batch_size, configfile):
        logger.debug("Initializing %s: (model: '%s', batch_size: %s)",
                     self.__class__.__name__, model, batch_size)
        super().__init__(model, images, batch_size, configfile, feeder=Feeder)

    def training_loop(self, model_inputs, model_targets):
        """ Running training on a batch of images for each side.

        Triggered from the training cycle in :class:`scripts.train.Train`.

        Parameters
        ----------
        model_inputs: list
            List of input tensors to the model
        model_targets: list
            List of target tensors for the model

        Returns
        loss: list
            The loss values for a batch
        """
        return self._model.model.train_on_batch(model_inputs, y=model_targets)


class Feeder(_Feeder):
    """ Handles the processing of a Batch for training the model and generating samples.

    Parameters
    ----------
    images: dict
        The list of full paths to the training images for this :class:`_Feeder` for each side
    model: plugin from :mod:`plugins.train.model`
        The selected model that will be running this trainer
    batch_size: int
        The size of the batch to be processed for each side at each iteration
    config: :class:`lib.config.FaceswapConfig`
        The configuration for this trainer
    alignments: dict
        A dictionary containing landmarks and masks if these are required for training for each
        side
    """

    def _load_generator(self, output_index):
        """ Load the :class:`~lib.training_data.TrainingDataGenerator` for this feeder.

        Parameters
        ----------
        output_index: int
            The output index from the model to get output shapes for

        Returns
        -------
        :class:`~lib.training_data.TrainingDataGenerator`
            The training data generator
        """
        logger.debug("Loading generator: %s", output_index)
        input_size = K.int_shape(self._model._inputs["face"][output_index])[1]
        outputs = self._model._outputs["face"][output_index]
        outputs = outputs if isinstance(outputs, list) else [outputs]
        output_shapes = [K.int_shape(out)[1:] for out in outputs]
        logger.debug("input_size: %s, output_shapes: %s", input_size, output_shapes)
        generator = TrainingDataGenerator(input_size,
                                          output_shapes,
                                          self._model.coverage_ratio,
                                          not self._model.command_line_arguments.no_augment_color,
                                          self._model.command_line_arguments.no_flip,
                                          self._model.command_line_arguments.warp_to_landmarks,
                                          self._alignments,
                                          self._config)
        return generator

    def get_batch(self):
        """ Get the feed data and the targets for each training side for feeding into the model's
        train function.

        Returns
        -------
        model_inputs: list
            The inputs to the model for each side A and B
        model_targets: list
            The targets for the model for each side A and B
        """
        model_inputs = []
        model_targets = dict()
        # TODO Tag these outputs somehow
        for output in ("decoder_df_a", "decoder_df_b"):
            side = output[-1]
            batch = next(self._feeds[side])
            side_inputs = batch["feed"]
            side_targets = self._compile_mask_targets(batch["targets"],
                                                      batch["masks"],
                                                      batch.get("additional_masks", None))
            if self._model.config["learn_mask"]:
                side_targets = side_targets + [batch["masks"]]
            logger.trace("side: %s, input_shapes: %s, target_shapes: %s",
                         side, [i.shape for i in side_inputs], [i.shape for i in side_targets])
            if get_backend() == "amd":
                model_inputs.extend(side_inputs)
            else:
                model_inputs.append(side_inputs)
            model_targets[output] = side_targets

        if (self._model.config["face_style_power"] > 0.0 or
                self._model.config["bg_style_power"] > 0.0):
            b_targets = model_targets["decoder_df_b"][0]
            style_targets = b_targets[..., :3]
            b_mask = b_targets[..., 3][..., None]
            if self._model.config["face_style_power"] > 0.0:
                style_targets = np.concatenate((style_targets, b_mask), axis=-1)
            if self._model.config["bg_style_power"] > 0.0:
                style_targets = np.concatenate((style_targets, 1.0 - b_mask), axis=-1)
            logger.trace("style_targets %s", style_targets.shape)
            model_targets["decoder_df_b_1"] = style_targets

        return model_inputs, model_targets
