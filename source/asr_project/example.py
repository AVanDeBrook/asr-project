from models import PretrainedFineTunedJasper
from utils import transcribe_audio
from pprint import pprint


if __name__ == "__main__":
    # list audio files to transcribe here
    audio_files = []

    # if you would like to include more metadata than just the audio path, use a list of dictionaries like below
    # audio_files = [
    #     {
    #         # path to the audio file (string)
    #         "path": "",
    #         # duration of the audio to transcribe (float)
    #         "duration": float(0.0),
    #         # offset (or start) of the audio, if there is more than one sample in each audio file (float)
    #         "offset": float(0.0),
    #         # ground truth or manual transcription of the audio, if available (string)
    #         "ground_truth": "",
    #         # word error rate (as a ratio)
    #         "wer": float(0.0)
    #     }
    # ]

    # path to the `.nemo` checkpoint to load a model from
    checkpoint_path = ""

    # load the asr model from the checkpoint above
    model = PretrainedFineTunedJasper(checkpoint_path=checkpoint_path)

    model_transcriptions = []
    # iterate over audio files and transcribe
    for path in audio_files:
        audio_transcription = transcribe_audio(path=path, model=model)
        model_transcriptions.append(audio_transcription)

    pprint(model_transcriptions)
