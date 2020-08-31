#!/usr/bin/env python3
""" Original Trainer """

from ._base import TrainerBase


class Trainer(TrainerBase):
    """ The original training function """

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
