import logging
import os
import json
from logging import DEBUG, INFO
from typing import *

import matplotlib.pyplot as plt
from data import ATCCompleteData, ATCO2SimData, ATCOSimData, Data, ZCUATCDataset
from models import (
    Model,
    PretrainedFineTunedJasper,
    PretrainedFineTunedQuartzNet,
    PretrainedJasper,
    PretrainedQuartzNet,
    RandomInitCTC,
)
from numpy.random import default_rng

logger = logging.getLogger(__name__)

# root dataset paths corresponding to data analysis classes
datasets: Dict[str, Data] = {
    # TODO: find a way to sync file paths across computers (shell/env var, config file?)
    "/home/students/vandebra/programming/thesis_data/atc0_comp": ATCCompleteData,
    "/home/students/vandebra/programming/thesis_data/atcosim/": ATCOSimData,
    "/home/students/vandebra/programming/thesis_data/ATCO2-ASRdataset-v1_beta": ATCO2SimData,
    # "/home/students/vandebra/programming/thesis_data/ZCU_CZ_ATC": ZCUATCDataset,
}

# name, model, number of epochs to train
models: List[Tuple[str, Model, int]] = [
    # Jasper models
    ("jasper_pretrained.nemo", PretrainedJasper, None),
    ("jasper_finetuned.nemo", PretrainedFineTunedJasper, 100),
    # # QuartzNet models
    ("quartznet_pretrained.nemo", PretrainedQuartzNet, None),
    ("quartznet_finetuned.nemo", PretrainedFineTunedQuartzNet, 100),
    # "Out-of-the-Box" models
    ("ctc_randominit.nemo", RandomInitCTC, 300),
]

if __name__ == "__main__":
    # TODO: clean this main statement
    # random seed initialization (passed to data classes) for reproducible randomness
    # in results
    logger.setLevel(DEBUG)
    RANDOM_SEED: int = 1
    random = default_rng(RANDOM_SEED)

    plt.style.use("ggplot")

    dataset_info = {"dataset_info": []}
    data_objects: List[Data] = []

    for root_path, data_class in datasets.items():
        data_analysis: Data = data_class(data_root=root_path, random_seed=RANDOM_SEED)
        print(data_analysis.name)

        # parse transcripts in dataset
        data_analysis.parse_transcripts()

        # tokens in dataset
        token_freq = data_analysis.token_freq_analysis(normalize=True)

        dataset_info["dataset_info"].append(
            {
                "dataset_name": data_analysis.name,
                "samples": data_analysis.num_samples,
                "duration": data_analysis.duration,
                "total_tokens": data_analysis.total_tokens,
                "unique_tokens": data_analysis.unique_tokens,
            }
        )
        data_objects.append(data_analysis)

    with open("manifests/dataset_stats.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset_info, indent=1))

    for o in data_objects[1:]:
        # concatenate everything to first object
        data_objects[0].concat(o)

    print(f"Unique tokens: {data_objects[0].unique_tokens}")

    """ Model Training/Testing """

    # TODO
    model_wers: List[Tuple[str, float]] = []
    for name, model_class, epochs in models:
        # create model
        model: Model = model_class(checkpoint_name=name)

        # train
        if not os.path.exists(f"checkpoints/{name}") and epochs is not None:
            model.training_setup(
                training_manifest_path="manifests/train_manifest.json",
                validation_manifest_path="manifests/valid_manifest.json",
                accelerator="gpu",
                max_epochs=epochs,
            )
            model.fit()

        # test
        model.testing_setup(test_manifest_path="manifests/test_manifest.json")
        model_wers.append(tuple([name, model.test()]))

    print("WERs:")
    print("-----------")
    for name, wer in model_wers:
        print(f"{name}: {wer}")
