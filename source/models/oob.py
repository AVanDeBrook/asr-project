"""
# TODO: file description
"""
from typing import *

from models import Model
from nemo.collections.asr.models import EncDecCTCModel
from omegaconf import DictConfig


class RandomInitCTC(Model):
    def __init__(self, checkpoint_name: str = "none"):
        self._config = self.load_config(config_path="config/config.yaml")
        self._model = EncDecCTCModel(cfg=DictConfig(self._config["model"]))

        super(RandomInitCTC, self).__init__(checkpoint_name)
