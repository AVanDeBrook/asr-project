"""
This file implements three permutations of The Jasper ASR model. It is a Deep
Time-Delay Neural Network (TDNN) comprised of block of one-dimensional convolutional
layers.

All Jasper models contain B blocks and R sub-blocks. Each sub-block contains a
one-dimensional convolution, batch normalization, ReLU activation, and dropout.

This model was trained and tested on the LibriSpeech corpus (conversational english).
Pretrained model checkpoints are also trained/tested on the LibriSpeech dataset.

See class descriptions for implementation specific deviations.

References:
-----------
[0] https://arxiv.org/abs/1904.03288
"""
from models import Model
from nemo.collections.asr.models import EncDecCTCModel
from typing import *
from omegaconf import DictConfig

class PretrainedJasper(Model):
    """
    This class instantiates a pretrained Jasper model (without fine-tuning on
    additional data) and tests it on the designated test data to obtain a
    baseline WER.

    Model name: stt_en_jasper_10x5dr
    Tokenizer: none
    """

    def __init__(
        self,
        pretrained_model_name: str = "stt_en_jasper10x5dr",
        name: str = "none",
    ):
        # create model
        self._model = EncDecCTCModel.from_pretrained(model_name=pretrained_model_name)
        # call super constructor to finish initializing the object
        super(PretrainedJasper, self).__init__(name)

    def fit(self):
        """
        Overriding since the goal of this class is just to establish a baseline WER
        before fine tuning or training from scratch
        """
        pass


class PretrainedFineTunedJasper(Model):
    def __init__(
        self,
        pretrained_model_name: str = "stt_en_jasper10x5dr",
        name: str = "none",
    ):
        # call super constructor to initialize the object
        self._config = self.load_config(config_path="config/jasper_10x5dr.yaml")

        self._config["model"]["train_ds"]["manifest_filepath"] = "manifests/train_manifest.json"
        self._config["model"]["validation_ds"]["manifest_filepath"] = "manifests/validation_manifest.json"

        self._model = EncDecCTCModel.from_pretrained(model_name=pretrained_model_name)

        # setup the model for training (dataloaders and pl trainer)
        self.setup(
            training_manifest_path="manifests/train_manifest.json",
            testing_manifest_path="manifests/test_manifest.json",
            validation_manifest_path="manifests/validation_manifest.json",
            accelerator="gpu",
            max_epochs=300,
        )

        super(PretrainedFineTunedJasper, self).__init__(name)


class RandomInitJasper(Model):
    def __init__(self, name: str = "none"):
        self._config = self.load_config(config_path="config/jasper_10x5dr.yaml")
        self._config["model"]["train_ds"]["manifest_filepath"] = "manifests/train_manifest.json"
        self._config["model"]["validation_ds"]["manifest_filepath"] = "manifests/validation_manifest.json"

        self._model = EncDecCTCModel(cfg=DictConfig(self._config["model"]))

        # setup the model for training (dataloaders and pl trainer)
        self.setup(
            training_manifest_path="manifests/train_manifest.json",
            testing_manifest_path="manifests/test_manifest.json",
            validation_manifest_path="manifests/validation_manifest.json",
            accelerator="gpu",
            max_epochs=300,
        )

        super(RandomInitJasper, self).__init__(name)