import torch
import gc
from nemo.collections.asr.models import EncDecCTCModel

if __name__ == "__main__":
    model = EncDecCTCModel.restore_from("checkpoints/ctc_randominit.nemo")
    model.setup_test_data(
        {
            "manifest_filepath": "manifests/test_manifest.json",
            "sample_rate": 16000,
            "labels": [
                " ",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
                "'",
            ],
            "batch_size": 8,
            "shuffle": False,
            "num_workers": 8,
            "pin_memory": True,
        }
    )

    # word error rate is defined as:
    # (substitutions + deletions + insertions) / number of words in label
    # i.e. (S+D+I) / N
    # S + D + I
    all_nums = []
    # N
    all_denoms = []

    # loop through test samples/batches and calculate individual WERs
    for test_batch in model.test_dataloader():
        # test batches are made up of the following:
        # [signal, signal length, target, target length]
        test_batch = [x.cuda() for x in test_batch]

        # get model predictions for test samples (don't care about any other
        # returned values at this point in time)
        _, _, predictions = model(
            input_signal=test_batch[0], input_signal_length=test_batch[1]
        )

        # calculate WER for this batch of predictions
        model._wer.update(
            predictions=predictions,
            targets=test_batch[2],
            target_lengths=test_batch[3],
        )

        # get WER from module (returns average, numerators, and denominators)
        _, nums, denoms = model._wer.compute()
        model._wer.reset()

        all_nums.append(nums.cpu().numpy())
        all_denoms.append(denoms.cpu().numpy())

        # clean up memory for next batch
        del test_batch, predictions, nums, denoms
        gc.collect()
        torch.cuda.empty_cache()

    print(float(sum(all_nums) / sum(all_denoms)))
