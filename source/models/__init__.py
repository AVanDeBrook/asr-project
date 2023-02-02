import logging
import torch
from model import Model
# from transformers import *

logger = logging.getLogger(__name__)

# hardware sanity check
if not torch.cuda.is_available():
    logger.warning(f"GPU or TPU is not detected or not available")
