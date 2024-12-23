import json
import os

from sacremoses import MosesDetokenizer
from typing import Optional

# Determine the project root programmatically
# (NOTE: feel free to set this manually as well)
PROJECT_ROOT = os.path.dirname(os.path.abspath("pyproject.toml"))

# Then we can set data paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
TRAIN_PATH = os.path.join(DATA_PATH, "train.json")
DEV_PATH = os.path.join(DATA_PATH, "dev.json")
TEST_PATH = os.path.join(DATA_PATH, "test.json")
ONTOLOGY_PATH = os.path.join(DATA_PATH, "ontology.json")
SPLIT_TO_PATH = {
    "train": TRAIN_PATH,
    "dev": DEV_PATH,
    "test": TEST_PATH,
}

# Paths to paraphrased source documents
PARAPHRASE_TYPES = frozenset({"blog", "book", "news", "radio", "reddit"})
PARAPHRASES_PATH = os.path.join(DATA_PATH, "seamus_paraphrases")
PARAPHRASES_TRAIN_PATH = os.path.join(PARAPHRASES_PATH, "train")
PARAPHRASES_DEV_PATH = os.path.join(PARAPHRASES_PATH, "dev")
PARAPHRASES_TEST_PATH = os.path.join(PARAPHRASES_PATH, "test")
PARAPHRASES_SPLIT_TO_PATH = {
    "train": PARAPHRASES_TRAIN_PATH,
    "dev": PARAPHRASES_DEV_PATH,
    "test": PARAPHRASES_TEST_PATH,
}

# Must unzip saved_contexts.zip to use this
SAVED_CONTEXTS_PATH = os.path.join(PROJECT_ROOT, "resources", "saved_contexts")
SAVED_CONTEXTS_TRAIN_PATH = os.path.join(SAVED_CONTEXTS_PATH, "train")
SAVED_CONTEXTS_DEV_PATH = os.path.join(SAVED_CONTEXTS_PATH, "dev")
SAVED_CONTEXTS_TEST_PATH = os.path.join(SAVED_CONTEXTS_PATH, "test")
SAVED_CONTEXTS_SPLIT_TO_PATH = {
    "train": SAVED_CONTEXTS_TRAIN_PATH,
    "dev": SAVED_CONTEXTS_DEV_PATH,
    "test": SAVED_CONTEXTS_TEST_PATH,
}

# Must unzip saved_prompts.zip to use this
SAVED_PROMPTS_PATH = os.path.join(PROJECT_ROOT, "resources", "saved_prompts")

# Some other helpful constants
SPLITS = frozenset({"train", "dev", "test"})
TEXT_FIELDS = frozenset({"report", "source", "report_summary", "combined_summary"})
ID_TYPES = frozenset({"seamus", "megawika"})

# We use this a lot
DETOKENIZER = MosesDetokenizer(lang="en")


# Mappings from MegaWika-style example IDs to
# FAMuS/SEAMuS-style IDs and vice-versa.
#
# The MegaWika-to-SEAMuS mapping is non-trivial,
# since SEAMuS includes FrameNet frame identifiers
# in the example IDs, whereas MegaWika does not---
# hence the need to construct this mapping here.
def seamus_key_to_megawika_key(
    seamus_key: str, paraphrase_type: Optional[str] = None
) -> str:
    """Convert a SEAMuS-style ID to a MegaWika-style one

    :param seamus_key: the SEAMuS ID
    :param paraphrase_type: a paraphrase type to be included
        in the output MegaWika ID
    """
    megawika_key = "-".join(seamus_key.split("-")[:3]).lower()
    if paraphrase_type is not None:
        megawika_key = megawika_key + "-" + paraphrase_type
    return megawika_key


def is_seamus_style_id(some_id: str) -> bool:
    """Determine whether a string is a SEAMuS-style identifier
    :param some_id: the input identifier
    :return: A bool indicating whether this is a SEAMuS-style identifier
    """
    return "frame" in some_id


SEAMUS_KEY_TO_MEGAWIKA_KEY = {ptype: dict() for ptype in PARAPHRASE_TYPES}
MEGAWIKA_KEY_TO_SEAMUS_KEY = {ptype: dict() for ptype in PARAPHRASE_TYPES}
for split in SPLITS:
    seamus_path = SPLIT_TO_PATH[split]
    with open(seamus_path) as f:
        seamus = json.load(f)
    for ptype in PARAPHRASE_TYPES:
        for ex in seamus:
            seamus_key = ex["instance_id"]
            megawika_key = seamus_key_to_megawika_key(seamus_key, ptype)
            SEAMUS_KEY_TO_MEGAWIKA_KEY[ptype][seamus_key] = megawika_key
            MEGAWIKA_KEY_TO_SEAMUS_KEY[ptype][megawika_key] = seamus_key
