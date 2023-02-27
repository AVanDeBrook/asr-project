import torch
import os
import librosa
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from typing import *


@torch.no_grad()
def transcribe_audio(
    path: str,
    model: ASRModel,
    duration: float = 0.0,
    offset: float = 0.0,
    device: Union[Literal["cuda"], Literal["cpu"]] = "cuda",
):
    assert path is not None
    assert os.path.exists(path)
    assert model is not None

    # save model states/values
    model_state = model.training
    dither_value = model.preprocessor.featurizer.dither
    pad_value = model.preprocessor.featurizer.pad_to

    # eliminate intentional randomness in preprocessing
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    # inference setup: put model in evaluation mode, freeze encoder/decoder
    model.eval()
    model.encoder.freeze()
    model.decoder.freeze()

    model.to(device)

    # get input data and length
    signal = AudioSegment.from_file(
        path, 16000, offset=offset, duration=duration
    ).samples

    # get model predictions/logits
    logits, logits_length, predictions = model.forward(
        input_signal=torch.tensor([signal]).to(device),
        input_signal_length=torch.tensor([signal.shape[0]]).long().to(device),
    )

    prediction, _ = model.decoding.ctc_decoder_predictions_tensor(logits, logits_length)

    # reset model states/preprocessor values
    model.train(mode=model_state)
    model.preprocessor.featurizer.dither = dither_value
    model.preprocessor.featurizer.pad_to = pad_value

    return prediction[0]
