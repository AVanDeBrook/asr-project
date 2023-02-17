import logging

import torch

from .jasper import PretrainedFineTunedJasper, PretrainedJasper
from .model import Model
from .oob import RandomInitCTC
from .quartznet import PretrainedFineTunedQuartzNet, PretrainedQuartzNet

# from .contextnet import
# from transformers import *

logger = logging.getLogger(__name__)

# hardware check
if not torch.cuda.is_available():
    logger.warning(f"GPU or TPU is not detected or not available")
