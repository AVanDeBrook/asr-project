from data import *

import json
import os
from typing import *

datasets: Dict[str, Data] = {
    "/home/avandebrook/programming/thesis_data/atc0_comp": ATCCompleteData,
    "/home/avandebrook/programming/thesis_data/atcosim/": ATCOSimData,
    "/home/avandebrook/programming/thesis_data/ATCO2-ASRdataset-v1_beta": ATCO2SimData,
    "/home/avandebrook/programming/thesis_data/ZCU_CZ_ATC": ZCUATCDataset,
}

if __name__ == "__main__":
    dataset_info = {"dataset_info": []}
    data_objects: List[Data] = []

    for root_path, data_class in datasets.items():
        data_analysis: Data = data_class(data_root=root_path)
        print(data_analysis.name)

        # parse transcripts in dataset
        data_analysis.parse_transcripts()
        print(f"Number of samples in {data_analysis.name}: {data_analysis.num_samples}")

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

    os.makedirs("manifests", exist_ok=True)
    with open("manifests/dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=4)

    # concatenate everything to first object
    for o in data_objects[1:]:
        data_objects[0].concat(o)

    dataset_all = data_objects[0]

    print(f"Trimmed duration: {dataset_all.duration}")
    print(f"Total samples: {dataset_all.num_samples}")
