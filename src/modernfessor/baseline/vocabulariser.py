from typing import Dict, Tuple, Union, List, Iterable
from pathlib import Path
from enum import Enum
from collections import defaultdict, OrderedDict

import json
from tqdm.auto import tqdm

# Core libraries
from tktkt.interfaces.vocabulariser import (
    Vocabulariser, Preprocessor)


class MorfessorVocabulariser(Vocabulariser):
    def __init__(
            self, preprocessor: Preprocessor,
            ):
        super().__init__(name="bpe", preprocessor=preprocessor)
