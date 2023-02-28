import gc
import logging
import os
from copy import deepcopy
from typing import *

import pytorch_lightning as pl
import torch
import yaml
from nemo.collections.asr.models import EncDecCTCModel
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

test_config = {
    # this field must be updated by the setup method
    "manifest_filepath": None,
    "sample_rate": 16000,
    "labels": [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
    ],
    "batch_size": 8,
    "shuffle": False,
    "num_workers": 8,
    "pin_memory": True,
}


class Model(object):
    """
    Wrapper class for automating training and comparing multiple models. Will
    also help keep things consistent if different APIs/frameworks are used.

    Attributes:
    -----------
    `_model`
    """

    def __init__(self, checkpoint_name: str = "none", model_class=EncDecCTCModel):
        if checkpoint_name == "none":
            logger.warning("Model name has been left to default.")

        self.checkpoint_name = checkpoint_name
        if os.path.exists(self.checkpoint_name):
            self._model = model_class.restore_from(self.checkpoint_name)

    def load_config(self, config_path: str) -> Dict:
        """
        Loads the model config file from the specified path.

        Arguments:
        ----------
        `config_path`: Path (relative or absolute) to the config file.

        Returns:
        --------
        `DictConfig`: Resulting dictionary (from loading YAML file) as
        an `omegaconf.DictCOnfig` object.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file path '{config_path}' does not exist"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(stream=f)

        return config

    def training_setup(
        self, training_manifest_path: str, validation_manifest_path: str, **trainer_args
    ) -> None:
        """
        Checks for valid manifest paths and sets up data loaders for the training,
        testing, and validation datasets.

        Sets up a pytorch lightning trainer for

        Arguments:
        ----------
        `training_manifest_path`: Path to the training manifest

        `testing_manifest_path`: Path to the testing manifest

        `validation_manifest_path`: Path to the validation manifest
        """
        assert self._config is not None
        if not os.path.exists(training_manifest_path):
            raise FileNotFoundError(
                f"Training manifest path '{training_manifest_path}' does not exist"
            )
        if not os.path.exists(validation_manifest_path):
            raise FileNotFoundError(
                f"Validation manifest path '{validation_manifest_path}' does not exist"
            )

        # specify manifest paths
        self._config["model"]["train_ds"]["manifest_filepath"] = training_manifest_path
        self._config["model"]["validation_ds"][
            "manifest_filepath"
        ] = validation_manifest_path

        # set up data partitions
        self._model.setup_training_data(DictConfig(self._config["model"]["train_ds"]))
        self._model.setup_validation_data(
            DictConfig(self._config["model"]["validation_ds"])
        )

        # initialize lightning trainer
        self._trainer = pl.Trainer(**trainer_args)

    def testing_setup(self, test_manifest_path: str):
        if not os.path.exists(test_manifest_path):
            raise FileNotFoundError(
                f"Test manifest path '{test_manifest_path}' does not exist"
            )

        # set manifest path
        test_config["manifest_filepath"] = test_manifest_path

        # set up test data partition
        self._model.setup_test_data(DictConfig(test_config))

    def fit(self) -> None:
        """
        Start the training process (for a NeMo model).
        """
        self._trainer.fit(self._model)

        os.makedirs("checkpoints", exist_ok=True)
        self._model.save_to(os.path.join("checkpoints", self.checkpoint_name))

    def test(
        self,
        testing_set: Union[Literal["train"], Literal["test"]] = "test",
        log_prediction: bool = False,
    ) -> float:
        """
        Tests the model and finds the average word error rate (WER) over test samples.

        Arguments:
        ----------
        `testing_set`: the dataset on which to test the model; can be either
        'train' or 'test'. Defaults to 'test'.

        Returns:
        --------
        `float`: Average WER over the test set.
        """
        # log test predictions if set
        self._model._wer.log_prediction = log_prediction
        self._model.cuda()
        self._model.eval()
        # word error rate is defined as:
        # (substitutions + deletions + insertions) / number of words in label
        # i.e. (S+D+I) / N
        # S + D + I
        all_nums = []
        # N
        all_denoms = []

        # loop through test samples/batches and calculate individual WERs
        for test_batch in self._model.test_dataloader():
            # test batches are made up of the following:
            # [signal, signal length, target, target length]
            test_batch = [x.cuda() for x in test_batch]

            # get model predictions for test samples (don't care about any other
            # returned values at this point in time)
            _, _, predictions = self._model(
                input_signal=test_batch[0], input_signal_length=test_batch[1]
            )

            # calculate WER for this batch of predictions
            self._model._wer.update(
                predictions=predictions,
                targets=test_batch[2],
                target_lengths=test_batch[3],
            )

            # get WER from module (returns average, numerators, and denominators)
            _, nums, denoms = self._model._wer.compute()
            self._model._wer.reset()

            all_nums.append(nums.cpu().numpy())
            all_denoms.append(denoms.cpu().numpy())

            # clean up memory for next batch
            del test_batch, predictions, nums, denoms
            gc.collect()
            torch.cuda.empty_cache()

        # return average over the dataset
        return sum(all_nums) / sum(all_denoms)

    @property
    def name(self) -> str:
        return self.checkpoint_name
