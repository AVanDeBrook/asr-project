from typing import *
import logging

logger = logging.getLogger(__name__)

class Model(object):
    """
    Wrapper class for automating training and comparing multiple models. Will
    also help keep things consistent if different APIs/frameworks are used.

    Attributes:
    -----------
    `_model`
    """
    _model = None

    def __init__(self, name: str="none"):
        if name == "none":
            logger.warning("Model name has been left to default.")

        assert self._model is not None

    def setup():
        raise NotImplementedError()

    def fit():
        raise NotImplementedError()

    def test():
        raise NotImplementedError()
