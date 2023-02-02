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
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from omegaconf import DictConfig
from model import Model
from nemo.collections.asr.models import EncDecCTCModel
from typing import *

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
        training_manifest_path: str,
        testing_manifest_path: str,
        validation_manifest_path: str,
        pretrained_model_name: str = "stt_en_jasper_10x5dr",
        name: str = "none"
    ):
        # create model
        self._model = EncDecCTCModel.from_pretrained(model_name=pretrained_model_name)
        # call super constructor to finish initializing the object
        super(PretrainedJasper, self).__init__(name)

    def setup():
        # get config; modify if needed
        pass

    def fit():
        pass

    def test():
        pass

class PretrainedFineTunedJasper(PretrainedJasper):
    def __init__(
        self,
        training_manifest_path: str,
        testing_manifest_path: str,
        validation_manifest_path: str,
        pretrained_model_name: str = "stt_en_jasper_10x5dr",
        name: str = "none"
    ):
        super(PretrainedFineTunedJasper, self).__init__(pretrained_model_name, name)

    def setup():
        pass

    def fit():
        pass

    def test():
        pass

class RandomInitJasper(Model):
    def __init__(
        self,
        config: Union
        training_manifest_path: str,
        testing_manifest_path: str,
        validation_manifest_path: str,
        name: str = "none"
    ):
        self._model = EncDecCTCModel()
        super(RandomInitJasper, self).__init__(name)

    def setup():
        pass

    def fit():
        pass

    def test():
        pass