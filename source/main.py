import json
import matplotlib.pyplot as plt
from typing import *
from data import Data, ATCCompleteData, ATCOSimData, ATCO2SimData, ZCUATCDataset
from matplotlib.figure import Figure

if __name__ == "__main__":
    # random seed initialization (passed to data classes) for reproducible randomness
    # in results
    RANDOM_SEED: int = 1

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
            data_analysis.dump_manifest(f"{data_analysis.name.replace(' ', '_')}_manifest.json", return_list=True)
        )

    # dump to a manifest
    with open("all_manifest.json", "w", encoding="utf-8") as f:
        for entry in concat_dataset:
            f.write(json.dumps(entry))
            f.write("\n")

    print(f"Total samples: {len(concat_dataset)}")
    print(f"Total unique tokens (across all datasets): {num_unique_tokens}")
    print(f"Total tokens: {num_tokens}")
