from typing import Dict, Tuple, Union, List, Iterable, Sequence
from pathlib import Path
from enum import Enum
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict

import json
from tqdm.auto import tqdm

# Core libraries
from tktkt.interfaces.vocabulariser import (
    Vocabulariser, Preprocessor,  NamedIterable, UnidentifiedVocab)

from morfessor.models.baseline import BaselineModel, BaseConstructionMethods
from morfessor.util.data import (
    freq_threshold, count_modifier, DataPoint, merge_counts, rand_split)
from morfessor.util.io import MorfessorIO


@dataclass
class MorfessorConfig():
    corpusweight: float
    use_skips: bool
    force_splits: Sequence[str] | None
    nosplit_re: str | None
    use_em: bool = False
    nolexcost: float
    freq_distr: str = "baseline"


class MorfessorVocabulariser(Vocabulariser):
    def __init__(
            self, preprocessor: Preprocessor,
            morfessor_config: MorfessorConfig,
            algorithms: Sequence[str],
            viterbismooth: float,
            viterbimaxlen: int,
            finish_threshold: float,
            maxepochs: int,
            freqthreshold: int,
            splitprob: float):
        super().__init__(name="morfessor", preprocessor=preprocessor)

        self.morfessor_config = morfessor_config
        self.model = BaselineModel(**asdict(morfessor_config))

        self.algorithms = list(algorithms)
        self.viterbismooth = viterbismooth
        self.viterbimaxlen = viterbimaxlen
        self.finish_threshold = finish_threshold
        self.maxepochs = maxepochs
        self.freqthreshold = freqthreshold
        self.splitprob = splitprob
        
        self.morfessor_io = MorfessorIO()

    def _vocabulariseFromWords(
            self, word_iterable: NamedIterable[Tuple[str, int]]) -> Path:
        out_folder = self._makeOutputFolder(word_iterable.name)

        dampfunc = None

        # Prep data
        data = [DataPoint(d[0], d[1], ()) for d in word_iterable]
        data = merge_counts(data)

        if self.freqthreshold > 1:
            data = freq_threshold(data, self.freqthreshold, False)
        if dampfunc is not None:
            data = count_modifier(data, dampfunc, False)
        if self.splitprob is not None:
            data = rand_split(data, BaseConstructionMethods, self.splitprob)

        # Note: The cases that preprocess beforehand do this because
        # the trainers only accept sentences. The other cases
        # do preprocessing internally.
        start_corpus_weight = self.model.get_corpus_coding_weight()

        # Set algorithm parameters
        if len(self.algorithms) == 0:
            self.algorithms.append('recursive')
        algparams = []
        for alg in self.algorithms:
            if alg == 'viterbi':
                algparams.append((self.viterbismooth, self.viterbimaxlen))
            else:
                algparams.append(())

        c = self.model.load_data(data)
        for alg, algp in zip(self.algorithms, algparams):
            e, c = self.model.train_batch(
                alg, algp, self.finish_threshold, self.maxepochs)

        self.write_segmentation_file(
            out_folder,
            self.model.get_segmentations())

        return out_folder

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> Path:
        """
        HuggingFace equivalent. For German: starts out extremely slow
        (giving an ETA of 500 000 hours), but finishes in under 2 hours.
        """
        out_folder = self._makeOutputFolder(sentence_iterable.name)

        dampfunc = None

        # Prep data
        data = list()
        for compound in sentence_iterable:
            if len(compound) > 0:
                data.append((1, self.morfessor_io._split_atoms(compound)))

        data = [DataPoint(d[0], d[1], ()) for d in data]
        data = merge_counts(data)

        if self.freqthreshold > 1:
            data = freq_threshold(data, self.freqthreshold, False)
        if dampfunc is not None:
            data = count_modifier(data, dampfunc, False)
        if self.splitprob is not None:
            data = rand_split(data, BaseConstructionMethods, self.splitprob)

        # Note: The cases that preprocess beforehand do this because
        # the trainers only accept sentences. The other cases
        # do preprocessing internally.
        start_corpus_weight = self.model.get_corpus_coding_weight()

        # Set algorithm parameters
        if len(self.algorithms) == 0:
            self.algorithms.append('recursive')
        algparams = []
        for alg in self.algorithms:
            if alg == 'viterbi':
                algparams.append((self.viterbismooth, self.viterbimaxlen))
            else:
                algparams.append(())

        c = self.model.load_data(data)
        for alg, algp in zip(self.algorithms, algparams):
            e, c = self.model.train_batch(
                alg, algp, self.finish_threshold, self.maxepochs)

        self.write_files(
            out_folder,
            self.model.get_segmentations())

        return out_folder

    def write_files(
            self, folder: Path, segmentations, **kwargs):
        """Write segmentation file.

        File format:
        <count> <construction1><sep><construction2><sep>...<constructionN>

        """

        self.morfessor_io.write_segmentation_file(
            folder / "merges.txt", segmentations)

        with open(folder / "vocab.json", 'w') as f:
            json.dump(asdict(self.morfessor_config), f)

    def _load(self):
        raise NotImplementedError()
