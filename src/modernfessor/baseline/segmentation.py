from typing import List, Mapping, Optional

from morfessor.models.baseline import BaselineModel, BaseConstructionMethods
from morfessor.util.io import MorfessorIO

from tktkt.interfaces.huggingface import (
    HuggingFaceTokeniserInterface)

from .vocabulariser import MorfessorConfig


class MorfessorBaseline(HuggingFaceTokeniserInterface):
    model: BaselineModel
    io: MorfessorIO
    viterbismooth: float
    viterbimaxlen: float

    def _tokenize(self, text, **kwargs) -> list[str]:
        atoms = list()
        for compound in self.io.compound_sep_re.split(text):
            if len(compound) > 0:
                atoms.append((1, self.io._split_atoms(compound)))

        constructions, logp = self.model.viterbi_segment(
            atoms, self.viterbismooth, self.viterbimaxlen)
        return constructions

    # The following three methods are for interfacing with the vocabulary.

    @property  # Property because that's how HuggingFace does it. Makes no sense to have getter/setter for this, but ok.
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def _convert_token_to_id(self, token: str) -> int:
        pass

    @abstractmethod
    def _convert_id_to_token(self, index: int) -> str:
        pass

    # The following two methods are for storage and come from the parent class of PreTrainedTokenizer.

    def get_vocab(self) -> Mapping[str, int]:
        compounds = self.model.get_compounds()

        atoms: list[str] = list()
        for c in compounds:

            _, _, splitloc = self.model._tree[c]
            constructions = []
            if not splitloc:
                atoms.append(c)
                constructions.append(c)

        # use stable mapping for simplicity now
        return {a: i for i, a in enumerate(atoms)}

    @abstractmethod
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        pass

    # The following methods are technically already implemented in HF, but it's important to define them explicitly.

    def tokenize(self, text: str, **kwargs) -> List[str]:
        return self._tokenize(text, **kwargs)

    @abstractmethod
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        pass

    @abstractmethod
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