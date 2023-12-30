# Environment Installation and Setup

## Conda Setup
```bash
conda create -n nemo python=3.8
conda activate nemo
```

## PyTorch and Cuda
Double-check the [PyTorch Getting Started page](https://pytorch.org/get-started/locally/) to make sure the most up-to-date version is installed:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## NeMo Dependencies
Double-check [NeMo's readme](https://github.com/NVIDIA/NeMo) to make sure all required dependencies are installed.

Debian-based systems (e.g. Ubuntu, Debian, etc.):
```bash
sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg
```

RPM-based systems (e.g. RHEL, Fedora, CentOS, etc.; may require [RPM Fusion to be setup](https://rpmfusion.org/Configuration#Command_Line_Setup_using_rpm)):
```bash
sudo dnf install -y libsndfile ffmpeg
```

All systems, after installing `libsndfile` and `ffmpeg`:
```bash
pip install Cython
pip install nemo_toolkit[all]
```
**Note**: Depending on terminal/shell `nemo_toolkit` may need to be escaped as one of the following:
* `nemo_toolkit['all']`
* `nemo_toolkit\[all\]`
* `"nemo_toolkit[all]"`

# Usage

Note that all code needs to be written and run from `source/asr_project`. Add a new file in that folder to run individual experiments.

## Training, Testing, and Inference Usage

To instantiate a model, a subclass of `Model` can be used. For example, to instantiate a pretrained Jasper model, the following can be used:

```python
from models import PretrainedJasper

# Specify the checkpoint path/file that should be used
# if the checkpoint already exists, it will be used
model = PretrainedJasper(checkpoint_name="jasper_checkpoint.nemo")

# save model to checkpoint
model._model.save_to("jasper_checkpoint.nemo")
```
The following can be used to instantiate a pretrained model and fine-tune it on additional data:

```python
from models import PretrainedFineTunedJasper

# instantiate the model (automatically loads from the
# checkpoint if it already exists)
model = PretrainedFineTunedJasper(checkpoint_name="checkpoint.nemo")

# set up the training configuration
model.training_setup(
    training_manifest_path="train_manifest.json",
    validatiaon_manifest_path="validation_manifest.json",
    max_epochs=10
)

# start the training loop
model.fit()

# set up the testing configuration
model.testing_setup(test_manifest_path="test_manifest.json")

# run inference over the test set and get the average word error rate
wer = model.test()
print(wer)
```

To run the model on audio data without labels i.e., inference, use the `transcribe_audio` function like below:

```python
from models import PretrainedFineTunedJasper
from utils import transcribe_audio

# load the pretrained model from a file (assume the checkpoint already exists)
model = PretrainedFineTunedJasper(checkpoint_name="jasper_checkpoint.nemo")

# pass the model and audio file path (assume the file, audio.wav, exists in the
# same directory as this file). Note that this function uses greedy decoding.
transcription = transcribe_audio(
    model=model,
    path="audio.wav"
)
print(transcription)
```

## Implementing a New ASR Model Class
Specific ASR models are implemented as a wrapper around the `Model` class. These should be organized into files for the relevant architecture. For example, Jasper models are implemented in `models/jasper.py`.

The `PretrainedFineTunedJasper` class is implemented as follows:
```python
from models import Model
from nemo.collections.asr.model import EncDecCTCModel

class PretrainedFineTunedJasper(Model):
    def __init__(
        self,
        pretrained_model_name: str = "stt_en_jasper10x5dr",
        checkpoint_path: str = "none",
    ):
        """
        :param pretrained_model_name: name of the pretrained model to download from NGC.
        :param checkpoint_path: path to the model checkpoint for loading/saving.
        """
        # load the configuration using the help method below
        self._config = self.load_config(config_path="config/jasper_10x5dr.yaml")

        # call the super constructor to initialize the object
        super(PretrainedFineTunedJasper, self).__init__(checkpoint_path)

        # sanity check; if the model has not been loaded, then a model is
        # downloaded and instantiated from the pretrained checkpoint from NVIDIA
        if self._model is None:
            self._model = EncDecCTCModel.from_pretrained(pretrained_model_name)
```

For this case, all that needs to be defined is the constructor for the class (the `__init__` method). In the constructor, before the super constructor is called, use the `Model.load_config` helper method to load a model config (use the config in `config/config.yaml` as the default if one does not exist).
```python
self._config = self.load_config(config_path="config/config.yaml")
```

Next, call the super constructor for the model class, like in the code above.
```python
super(PretrainedFineTunedJasper, self).__init__(checkpoint_path)
```

This does most of the heavy lifting of loading and initializing the actual ASR model from NeMo. After the super constructor is called any user code can then be run. In this case, since this is class is supposed to use a pretrained model and fine-tune it on additional data, the code checks the `Model._model` variable, if it has not been set that means no existing checkpoint was found, so the pretrained checkpoint NVIDIA should be used:
```python
if self._model is None:
    self._model = EncDecCTCModel.from_pretrained(pretrained_model_name)
```
