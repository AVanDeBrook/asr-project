"""
This file implements two permutations of the Quartznet model. It is an extension
of Jasper with separable convolutions and larger filters. Only two are trained since
the third permutation would be a randomly initialized model identical to Jasper.

QuartzNet models are made up of blocks and convolutional sub-blocks (B and R,
respectively). Each sub-block contains a one-dimensional separable convolution,
batch normalization, ReLU activation and dropout.

QuartzNet was trained and tested on six datasets: LibriSpeech, Mozilla Common Voice,
Wall Street Journal, Fisher, Switchboard, and NSC Singapore English (all conversational
English).

See class descriptions for implementation specific deviations.

References:
-----------
[0] https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#quartznet
[1] https://arxiv.org/abs/1910.10261
[2] https://catalog.ngc.nvidia.com/orgs/nvidia/models/nemospeechmodels
"""
from copy import deepcopy
from typing import *

from models import Model
from nemo.collections.asr.models import EncDecCTCModel
from omegaconf import DictConfig


class PretrainedQuartzNet(Model):
    """
    This class instantiates a pretrained QuartzNet model (without fine-tuning on
    additional data) and tests it on the designated test data to obtain a
    baseline WER).

    Model name: QuartzNet15x5Base-En
    Tokenizer: none
    """

    def __init__(
        self,
        pretrained_model_name: str = "QuartzNet15x5Base-En",
        checkpoint_name: str = "none",
    ):
        self._config = self.load_config(config_path="config/quartznet_15x5.yaml")
        super(PretrainedQuartzNet, self).__init__(checkpoint_name)
        if self._model is None:
            self._model = EncDecCTCModel.from_pretrained(pretrained_model_name)

    def fit(self):
        """
        Overriding since the goal of this class is just to establish a baseline
        WER before fine tuning or training from scratch.
        """
        pass


class PretrainedFineTunedQuartzNet(Model):
    def __init__(
        self,
        pretrained_model_name: str = "QuartzNet15x5Base-En",
        checkpoint_name: str = "none",
    ):
        self._config = self.load_config(config_path="config/quartznet_15x5.yaml")
        super(PretrainedFineTunedQuartzNet, self).__init__(checkpoint_name)
        if self._model is None:
            self._model = EncDecCTCModel.from_pretrained(pretrained_model_name)
