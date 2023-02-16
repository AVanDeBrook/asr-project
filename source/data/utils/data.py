import json
import os
from collections import namedtuple
from pathlib import Path
from typing import *

import matplotlib
import matplotlib.collections
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


class Data:
    """
    Top-level Data class that provides several methods for processing and analyzing text datasets for NLP processes.

    This class should be extended and the following methods/properties implemented for each dataset:
    * `parse_transcripts`
    * `name`

    Attributes:
    -----------
    `data`: list of dictionary objects. Each object corresponds to one data sample
    and typically contains the following metadata:
    * `audio_filepath` (required) - path to the audio data (input data). Type: `str`,
    absolute file path, conforms to `os.PathLike`
    * `duration` (required) - duration, in seconds, of the audio data. Type: `float`
    * `text` (required) - transcript of the audio data (label/ground truth). Type: `str`
    * `offset` - if more than one sample is present in a single audio file, this field
    specifies its offset i.e. start time in the audio file. Type: `float`

    `_random`: numpy seeded RNG instance

    `_normalized`: bool indicating whether samples in the dataset have been normalized/preprocessed
    """

    data: List[Dict[str, Union[float, str]]]
    _random: np.random.Generator
    _normalized: bool

    def __init__(self, data_root: str, random_seed: int = None):
        """
        Arguments:
        ----------
        `data_root`: path to the base of the dataset, basically just a path from which the
        audio and transcript data can be found. Varies by dataset and implementation.
        """
        assert isinstance(data_root, str)
        assert os.path.exists(data_root), f"Path does not exist {data_root}"

        # create random number generator sequence with specified seed, if applicable
        Data._random = np.random.default_rng(random_seed)

    def parse_transcripts(self) -> List[str]:
        """
        This method must be overridden and implemented for each implementation of this class
        for datasets.

        Returns:
        --------
        Dictionary (from `json` module) with necessary data info e.g. annotations, file
        path, audio length, offset.
        """
        raise NotImplementedError(
            "This is an abstract method that should be implemented and overridden for "
            "all classes that implement this one. If this method has been called, there "
            "is an issue with the class that extended the Data class."
        )

    def create_token_hist(
        self,
        token_counts: List[int] = [],
    ) -> matplotlib.figure.Figure:
        """
        Calculates the number of utterances in each sample and generates a histogram.

        Utterance counts are determined by splitting each transcript on whitespace and
        calculating the length of the resulting list.

        TODO: add flexibility for plot types.
        TODO: issue with figures (figures/axes need to be cleared between runs)

        Arguments:
        ----------
        `utterance_counts`: (Optional) `list` of `ints` for precalculated utterance counts.

        `plot_type`: (Optional) `str` type of plot tool to use to create the histogram.
        Can be either `"seaborn"` or `"matplotlib"`. Defaults to `"seaborn"`.

        Returns:
        --------
        Either a `matplotlib.pyplot.Figure` or `seaborn.object.Plot` instance, depending on the value of `plot_type`.
        """
        # Clear figure and axes
        plt.clf(), plt.cla()
        # check if manifest data has been generated, parse transcripts and generate
        # manifest data if not
        if len(self.data) == 0:
            self.parse_transcripts()

        # check if utterance counts (optional arg) has been provided, calculate utterance
        # counts from transcriptions
        if len(token_counts) == 0:
            for data in self.data:
                data = data["text"]
                words = data.split(" ")
                token_counts.append(len(words))

        histogram = plt.hist(token_counts)
        plt.xlabel("Bins")
        plt.ylabel("Token Counts")
        plt.title(f"Tokens per Sample in {self.name}")

        return histogram

    def token_freq_analysis(self, normalize=False) -> Dict[str, Union[int, List]]:
        """
        Perform a token frequency analysis on the dataset (number of occurrences of each token throughout the dataset).

        Arguments:
        ----------
        `normalize`: (optional)`bool`, whether to normalize values such that all frequencies add to 1.

        Returns:
        --------
        `token_freqs`: `dict` with tokens and number of occurrences of those tokens throughout the dataset.
        If `normalize` is set to `True` a dictionary with tokens corresponding to a list is returned e.g.
        ```
        {
            "token": [24, 0.0486] # token with corresponding occurences and frequencies
        }
        ```

        """
        if len(self.data) == 0:
            self.parse_transcripts()

        # tokens corresponding to occurences/frequencies
        token_freqs = {}

        for sample in self.data:
            sample = sample["text"]
            for token in sample.split():
                if token in token_freqs.keys():
                    # increment occurences if there is already an entry
                    token_freqs[token] += 1
                else:
                    # add entry if one does not exist already
                    token_freqs[token] = 1

        if normalize:
            # convert from token occureces to frequencies: (# of token occurences / # of tokens)
            num_tokens = len(token_freqs)
            for token, freq in token_freqs.items():
                token_freqs[token] = [freq, float(freq) / num_tokens]

        return token_freqs

    def normalize_data(self):
        pass

    def dump_manifest(
        self, outfile: str, make_dirs: bool = True, return_list: bool = False
    ) -> Union[None, List[Dict]]:
        """
        Dump input data paths, labels, and metadata to `outfile` in NeMo manifest format.

        Arguments:
        ----------
        `outfile`: `str`, output path

        `make_dirs`: (optional) `bool`, whether to make nonexistent parent directories
        in `outfile`. Defaults to `True`.

        `return_list`: Returns a list of strings instead of creating and dumping to a file.

        Returns:
        --------
        None
        """
        outfile = Path(outfile).absolute()
        os.makedirs(str(outfile.parent), exist_ok=make_dirs)

        # check if manifest data has been generated
        if len(self.data) == 0:
            self.parse_transcripts()

        # write each data point its own line in the file, in json format (conform to NeMo
        # manifest specification)
        with open(str(outfile), "w") as manifest:
            for entry in self.data:
                manifest.write(json.dumps(entry))
                manifest.write("\n")

        if return_list:
            return self.data

    def concat(self, dataset: "Data"):
        self.data.extend(dataset.data)

    @property
    def name(self) -> str:
        """
        Name of this dataset
        """
        raise NotImplementedError(
            "This property should be implemented by the extending class"
        )

    @property
    def num_samples(self) -> int:
        """
        Number of samples in this dataset
        """
        return len(self.data)

    @property
    def hours(self) -> float:
        """
        Effective hours of data (silence removed)
        """
        total_hours = 0
        for item in self.data:
            total_hours += item["duration"]
        return total_hours

    @property
    def total_tokens(self) -> int:
        """
        Total number of tokens in the transcripts of the dataset (labels)
        """
        num_tokens = 0
        for item in self.data:
            tokens = item["text"].split(" ")
            num_tokens += len(tokens)
        return num_tokens

    @property
    def unique_tokens(self) -> int:
        """
        Number of unique tokens in the dataset. Repeating tokens are removed
        from this total
        """
        unique_tokens = []
        for item in self.data:
            tokens = item["text"].split(" ")
            for token in tokens:
                if token not in unique_tokens:
                    unique_tokens.append(token)
        return unique_tokens
