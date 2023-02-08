import json
import matplotlib.pyplot as plt
import logging
from logging import INFO
from numpy.random import default_rng
from typing import *
from data import Data, ATCCompleteData, ATCOSimData, ATCO2SimData, ZCUATCDataset
from models import (
    Model,
    PretrainedFineTunedJasper,
    PretrainedJasper,
    PretrainedFineTunedQuartzNet,
    PretrainedQuartzNet,
    RandomInitCTC,
)
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # TODO: clean this main statement
    # random seed initialization (passed to data classes) for reproducible randomness
    # in results
    logger.setLevel(INFO)
    RANDOM_SEED: int = 1
    random = default_rng(RANDOM_SEED)

    plt.style.use("ggplot")

    # root dataset paths corresponding to data analysis classes
    datasets: Dict[str, Data] = {
        # TODO: find a way to sync file paths across computers (shell/env var, config file?)
        "/home/students/vandebra/programming/thesis_data/atc0_comp": ATCCompleteData,
        "/home/students/vandebra/programming/thesis_data/atcosim/": ATCOSimData,
        "/home/students/vandebra/programming/thesis_data/ATCO2-ASRdataset-v1_beta": ATCO2SimData,
        # "/home/students/vandebra/programming/thesis_data/ZCU_CZ_ATC": ZCUATCDataset,
    }

    # datasets combined into one large dataset (also transformed into a common format)
    concat_dataset = []

    # find number of unique tokens across datasets
    num_unique_tokens = 0
    num_tokens = 0

    for root_path, data_class in datasets.items():
        data_analysis: Data = data_class(data_root=root_path, random_seed=RANDOM_SEED)
        print(data_analysis.name)

        # parse transcripts in dataset
        data_analysis.parse_transcripts()

        # tokens in dataset
        token_freq = data_analysis.token_freq_analysis(normalize=True)
        num_unique_tokens += len(token_freq.keys())
        for counts in token_freq.values():
            num_tokens += counts[0]

        # concatenate individual datasets into one larger dataset
        concat_dataset.extend(
            data_analysis.dump_manifest(
                f"manifests/{data_analysis.name.replace(' ', '_')}_manifest.json",
                return_list=True,
            )
        )

    # TODO: create a function for and clean up this mess
    # dump to a manifest
    with open("manifests/all_manifest.json", "w", encoding="utf-8") as f:
        for entry in concat_dataset:
            f.write(json.dumps(entry))
            f.write("\n")

    random.shuffle(concat_dataset)
    num_train_samples = int(len(concat_dataset) * 0.8)
    num_valid_samples = int(num_train_samples * 0.2)

    with open("manifests/train_manifest.json", "w", encoding="utf-8") as f:
        for entry in concat_dataset[: (num_train_samples - num_valid_samples)]:
            f.write(json.dumps(entry))
            f.write("\n")

    with open("manifests/validation_manifest.json", "w", encoding="utf-8") as f:
        for entry in concat_dataset[
            (num_train_samples - num_valid_samples) : num_train_samples
        ]:
            f.write(json.dumps(entry))
            f.write("\n")

    with open("manifests/test_manifest.json", "w", encoding="utf-8") as f:
        for entry in concat_dataset[num_train_samples:]:
            f.write(json.dumps(entry))
            f.write("\n")

    print(f"Total samples: {len(concat_dataset)}")
    print(f"Total unique tokens (across all datasets): {num_unique_tokens}")
    print(f"Total tokens: {num_tokens}")

    """ Model Training/Testing """

    # name, model pairs
    models: Dict[str, Model] = {
        # Jasper models
        "jasper_pretrained.nemo": PretrainedJasper,
        "jasper_finetuned.nemo": PretrainedFineTunedJasper,
        # QuartzNet models
        "quartznet_pretrained.nemo": PretrainedQuartzNet,
        "quartznet_finetuned.nemo": PretrainedFineTunedQuartzNet,
        # "Out-of-the-Box" models
        "ctc_randominit.nemo": RandomInitCTC,
    }

    # TODO
    model_wers = []
    for name, model in models.items():
        # create model
        model: Model = model(name=name)
        # train
        model.fit()
        # test
        model_wers.append(model.test())

    print("WERs:")
    print("-----------")
    for name, wer in zip(models.values(), model_wers):
        print(f"{name}: {wer}")
