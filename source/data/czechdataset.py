import os
import glob
import re
from typing import *
from data import Data
from xml.etree import ElementTree


class ZCUATCDataset(Data):
    """
    Data is organized into `trs` files (follows an XML standard). Details in `~parse_transcripts` function.

    Dataset can be obtained here: https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0001-CCA1-0

    Notes:
        - As useful as this dataset may be, the transcriptions and their format are so convoluted and different from
          the other datasets that it may not be worth using
        - There are a lot of non-english tokens in this data that I am not sure how to filter out
        - All number readouts are space separated
        - **I'll keep working on this script to normalize the data, but I'm exluding it from the training data for now**
        - Examples of problematic lines:
            * Austrian 1 2 8 X contact (Vienna(vín)) 1 3 1 . 3 5 0 servus
            * Air Malta (9(najn)) 6 9 1 contact Praha 1 2 7 . 1 2 5 go+
            * (Praha(Prague)) 1 3 5 1 3 5  7 9
            * +el 3 8 0 plea+
    """

    def __init__(self, data_root: str, **kwargs):
        super(ZCUATCDataset, self).__init__(dataset_name="ATCC", **kwargs)

        self.transcript_paths = glob.glob(os.path.join(data_root, "*.trs"))
        self.audio_paths = glob.glob(os.path.join(data_root, "*.wav"))

        assert (
            len(self.transcript_paths) != 0
        ), f"Cannot find transcripts in data_root: {data_root}"

        assert (
            len(self.audio_paths) != 0
        ), f"Cannot find audio files in data_root: {data_root}"

        assert len(self.transcript_paths) == len(
            self.audio_paths
        ), f"Number of audio files and transcripts does not match ({len(self.audio_paths)} and {len(self.transcript_paths)}, respectively)"

    def parse_transcripts(self) -> List[str]:
        """
        Since the transcript files correspond to wav files for ASR tasks, the transcriptions are organized into
        <Sync> elements with time attributes e.g. <Sync time="1.900"/>. In this case the time attribute is ignored
        and only the text is extracted. The node hierarchy is as follows:
            * <Trans> -- root node
            * <Episode> -- empty tag (organizational?)
            * <Section> -- metadata, similar to <Turn>
            * <Turn> -- contains metadata about the duration of the audio file (attributes), serves as parent for <Sync> nodes
            * <Sync> -- holds the time (attribute) and text info (tail)

        There are also transcriber annotations present in the text, usually following a form similar to other transcripts
        for example:
            * [air]
            * [ground]
            * [unintelligible]
        """
        data = []

        annotation_tag = re.compile(r"(\[[A-Za-z_\|\?]+\])")

        for audio_path, transcript_path in zip(self.audio_paths, self.transcript_paths):
            doc_data = []
            try:
                # root node: <Trans>
                document = ElementTree.parse(transcript_path).getroot()
            except ElementTree.ParseError as e:
                # because not all transcripts conform to the given format
                with open(transcript_path, "a") as dumb_path:
                    # there is a single file that is missing the closing tags on all nodes
                    # hard-coding the fix here, so I can forget about it
                    dumb_path.write("</Turn>\n</Section>\n</Episode>\n</Trans>\n")
                # parse the document again
                document = ElementTree.parse(transcript_path).getroot()

            # find total audio duration
            # start and end times are listed in the 'Section' node
            section_node = document.find(".//Section")
            audio_duration = float(section_node.attrib["endTime"])

            # find <Sync> tags, extract text, reformat/clean
            for sync_node in document.iterfind(
                ".//Sync"
            ):  # searches all subelements for Sync nodes
                text = annotation_tag.sub("", sync_node.tail).strip()
                # ".." corresponds to silent segments, "" can occur when the transcript is made up
                # of only transcriber annotations
                if text != ".." and text != "":
                    doc_data.append(
                        {
                            "audio_filepath": audio_path,
                            "text": text,
                            "offset": float(sync_node.attrib["time"]),
                        }
                    )

            # check to make sure useful samples were found in the document
            if len(doc_data) != 0:
                # determine durations for each sample in this document
                for i in range(len(doc_data) - 1):
                    doc_data[i]["duration"] = (
                        doc_data[i + 1]["offset"] - doc_data[i]["offset"]
                    )
                doc_data[-1]["duration"] = audio_duration - doc_data[-1]["offset"]

                data.extend(doc_data)

        self.data = data
        return data
