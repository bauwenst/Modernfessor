from typing import List, Mapping, Optional

from morfessor.models.baseline import MorfessorBaseline as MorfessorBackend, BaseConstructionMethods
from morfessor.util.io import MorfessorIO

from tktkt.interfaces.huggingface import HuggingFaceTokeniserInterface

from .vocabulariser import MorfessorConfig


class MorfessorBaseline(HuggingFaceTokeniserInterface):

    def __init__(self, backend: MorfessorBackend, io: MorfessorIO, viterbi_smooth: float, viterbi_maxlen: float, **kwargs):
        super().__init__(**kwargs)
        self._backend = backend
        self._io = io
        self._smoothing = viterbi_smooth
        self._max_len = viterbi_maxlen

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        atoms = []
        for compound in self._io.compound_sep_re.split(text):
            if compound:
                atoms.append((1, self._io._split_atoms(compound)))

        constructions, logp = self._backend.viterbi_segment(atoms, self._smoothing, self._max_len)
        return constructions

    # The following three methods are for interfacing with the vocabulary.

    @property  # Property because that's how HuggingFace does it. Makes no sense to have getter/setter for this, but ok.
    def vocab_size(self) -> int:
        pass

    def _convert_token_to_id(self, token: str) -> int:
        pass

    def _convert_id_to_token(self, index: int) -> str:
        pass

    # The following two methods are for storage and come from the parent class of PreTrainedTokenizer.

    def get_vocab(self) -> Mapping[str, int]:
        compounds = self._backend.get_compounds()

        atoms: list[str] = list()
        for c in compounds:

            _, _, splitloc = self._backend._tree[c]
            constructions = []
            if not splitloc:
                atoms.append(c)
                constructions.append(c)

        # use stable mapping for simplicity now
        return {a: i for i, a in enumerate(atoms)}

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        pass

    # The following methods are technically already implemented in HF, but it's important to define them explicitly.

    def tokenize(self, text: str, **kwargs) -> List[str]:
        return self._tokenize(text, **kwargs)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        pass

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        """
        Takes over the role of tokenizers.processors (adding [CLS] and [SEP]) in PreTrainedTokenizer, where it is called by:
            ._encode_plus()
                .tokenize(text)
                    ._tokenize(text)
                .convert_tokens_to_ids(tokens)
                .prepare_for_model(ids)
                    .build_inputs_with_special_tokens(ids)
                    .create_token_type_ids_from_sequences(...)
        """
        pass