import logging
import os
import pytorch_lightning as pl
from copy import deepcopy
from typing import *
from nemo.collections.asr.models import EncDecCTCModel
from omegaconf import DictConfig
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)


class Model(object):
    """
    Wrapper class for automating training and comparing multiple models. Will
    also help keep things consistent if different APIs/frameworks are used.

    Attributes:
    -----------
    `_model`
    """

    _model: EncDecCTCModel
    _config: DictConfig
    _trainer: pl.Trainer
    _no_training: bool
    _name: str

    def __init__(self, name: str = "none"):
        if name == "none":
            logger.warning("Model name has been left to default.")

        self._name = name

        # model should be set up by the implementing class
        assert self._model is not None

    def load_config(self, config_path: str) -> DictConfig:
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
        assert os.path.exists(
            config_path
        ), f"Could not find config file: '{config_path}'"

        with open(config_path, "r", encoding="utf-8") as f:
            config = YAML(typ="safe").load(f)

        self._config = DictConfig(config)
        return config

    def setup(
        self,
        training_manifest_path: str,
        testing_manifest_path: str,
        validation_manifest_path: str,
        **pl_args,
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
        # dataloaders
        self._setup_dataloaders(
            training_manifest_path, testing_manifest_path, validation_manifest_path
        )
        # pytorch lightning trainer
        self._setup_training(**pl_args)

    def fit(self) -> None:
        """
        Start the training process (for a NeMo model).
        """
        self._trainer.fit(self._model)

        os.makedirs("checkpoints", exist_ok=True)
        self._model.save_to(f"checkpoints/{self.name}.nemo")

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
            _, _, predictions = self._model.forward(
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

            all_nums.append(nums)
            all_denoms.append(denoms)

            # clean up memory for next batch
            del test_batch, predictions, nums, denoms

            # return average over the dataset
            return sum(all_nums) / sum(all_denoms)

    @property
    def name(self) -> str:
        return self._name

    def _setup_dataloaders(
        self,
        training_manifest_path: str,
        testing_manifest_path: str,
        validation_manifest_path: str,
    ) -> None:
        """
        Checks for valid manifest paths and sets up data loaders for the training,
        testing, and validation datasets.

        Arguments:
        ----------
        `training_manifest_path`: Path to the training manifest

        `testing_manifest_path`: Path to the testing manifest

        `validation_manifest_path`: Path to the validation manifest
        """
        assert os.path.exists(
            training_manifest_path
        ), f"Could not find training manifest path: '{training_manifest_path}'"
        assert os.path.exists(
            testing_manifest_path
        ), f"Could not find testing manifest path: '{testing_manifest_path}'"
        assert os.path.exists(
            validation_manifest_path
        ), f"Could not find validation manifest path: '{validation_manifest_path}'"

        # training config setup
        self._config["model"]["train_ds"]["manifest_filepath"] = training_manifest_path
        self._model.setup_training_data(self._config["model"]["train_ds"])

        # validation config setup
        self._config["model"]["validation_ds"][
            "manifest_filepath"
        ] = validation_manifest_path
        self._model.setup_validation_data(self._config["model"]["validation_ds"])

        # testing config setup
        self._config["model"]["test_ds"] = deepcopy(
            self._config["model"]["validation_ds"]
        )
        self._config["model"]["test_ds"]["manifest_filepath"] = testing_manifest_path
        self._model.setup_test_data(self._config["model"]["test_ds"])

    def _setup_training(self, **kwargs) -> None:
        """
        Sets up a pytorch lightning trainer from passed args.

        Arguments:
        ----------
        `**kwargs`: keyword arguments to be passed to the `pl.Trainer` constructor.
        """
        self._trainer = pl.Trainer(**kwargs)
