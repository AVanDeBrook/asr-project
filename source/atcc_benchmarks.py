import json
from models import PretrainedFineTunedJasper, RandomInitCTC
from data import Data, ATCCompleteData, split_data
from typing import List


def split_train_set(training_set: Data, split_ratio=0.2) -> List[Data]:
    train_data_splits = []
    # calculate how many iterations it will take to cover the entire dataset
    iterations = int(1.0 / split_ratio)

    current_split = split_ratio
    for i in range(iterations):
        # calculate length of this split
        data_length = int(len(training_set.data) * current_split)

        # add current split to return list
        train_data_splits.append(Data.from_iterable(training_set.data[:data_length]))

        # increment to next split
        current_split += split_ratio

    return train_data_splits


if __name__ == "__main__":
    results = {"results": []}
    # Path to root of ATCC
    atcc_path = "/home/students/vandebra/programming/thesis_data/atc0_comp"
    # initialize atcc dataset utility
    atcc = ATCCompleteData(data_root=atcc_path, random_seed=1)

    # parse the transcripts into a common format
    atcc.parse_transcripts()

    train, test = split_data(atcc, test=True, validation=False, test_split_ratio=0.2)
    train_splits = split_train_set(train, split_ratio=0.2)

    test.dump_manifest("atcc_benchmarks/atcc_test.json")

    for i, split in enumerate(train_splits):
        checkpoint_name = f"atcc_benchmarks/jasper_train_split={i}.nemo"
        train_manifest = f"atcc_benchmarks/atcc_train_split={i}.json"
        # dump training split to manifest
        split.dump_manifest(train_manifest)
        # create model
        model = PretrainedFineTunedJasper(checkpoint_name=checkpoint_name)

        # set up training data
        model.training_setup(
            training_manifest_path=train_manifest, accelerator="gpu", max_epochs=150,
        )

        # start training loop
        model.fit()

        model.testing_setup(test_manifest_path="atcc_benchmarks/atcc_test.json")

        test_wer = model.test(testing_set="test")
        train_wer = model.test(testing_set="train")

        results["results"].append(
            {
                checkpoint_name: {
                    "test_wer": test_wer,
                    "train_wer": train_wer,
                    "epochs": 150,
                    "training_manifest": train_manifest,
                    "test_manifest": "atcc_benchmarks/atcc_test.json",
                    "split_ratio": 0.2,
                }
            }
        )

    with open("atcc_benchmarks/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
