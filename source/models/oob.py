"""
# TODO: file description
"""
from models import Model
from nemo.collections.asr.models import EncDecCTCModel
from omegaconf import DictConfig
from typing import *


class RandomInitCTC(Model):
    def __init__(self, checkpoint_name: str = "none"):
        self._config = self.load_config(config_path="config/config.yaml")
        self._config["model"]["train_ds"][
            "manifest_filepath"
        ] = "manifests/train_manifest.json"
        self._config["model"]["validation_ds"][
            "manifest_filepath"
        ] = "manifests/validation_manifest.json"

        self._model = EncDecCTCModel(cfg=DictConfig(self._config["model"]))

        # setup the model for training (dataloaders and pl trainer)
        self.setup(
            training_manifest_path="manifests/train_manifest.json",
            testing_manifest_path="manifests/test_manifest.json",
            validation_manifest_path="manifests/validation_manifest.json",
            accelerator="gpu",
            max_epochs=300,
        )

        super(RandomInitCTC, self).__init__(checkpoint_name)
