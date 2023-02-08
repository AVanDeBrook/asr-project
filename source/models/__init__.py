import logging
import torch
from .model import Model
from .jasper import PretrainedJasper, PretrainedFineTunedJasper, RandomInitJasper

# from transformers import *

logger = logging.getLogger(__name__)

# hardware check
if not torch.cuda.is_available():
    logger.warning(f"GPU or TPU is not detected or not available")
