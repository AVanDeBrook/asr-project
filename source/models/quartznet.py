from model import Model
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl

class PretrainedQuartzNet(Model):
    """

    """
    def __init__(self, model, tokenizer, name: str = "none"):
        super().__init__(model, tokenizer, name)

    def setup():
        pass

    def fit():
        pass

    def test():
        pass

class PretrainedFineTunedQuartzNet(Model):
    def __init__(self, model, tokenizer, name: str = "none"):
        super().__init__(model, tokenizer, name)

    def setup():
        pass

    def fit():
        pass

    def test():
        pass

class RandomInitQuartzNet(Model):
    def __init__(self, model, tokenizer, name: str = "none"):
        super().__init__(model, tokenizer, name)

    def setup():
        pass

    def fit():
        pass

    def test():
        pass