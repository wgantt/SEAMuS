import json
import spacy

from datasets import Dataset
from functools import partial
from typing import Any, Dict, Iterator, Optional

from seamus.constants import SPLIT_TO_PATH

nlp = spacy.load("en_core_web_sm")


def gen(
    split: str,
    split_override: Optional[str] = None,
    source_context_override_path: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Generate examples from the FAMUS-SUM dataset

    :param split: The split to generate examples from
    :param split_override: Path to a JSON file containing examples to use
        in lieu of the original split.
    :param source_context_override_path: Path to a JSON file containing text
        to use in place of the original source document. For experiments, we
        want to use smaller selections from the source documents obtained via
        retrieval methods (see seamus/retrieval/)"""
    if split_override is not None:
        print(f"Using split override: {split_override}")
        data_path = split_override
    else:
        data_path = SPLIT_TO_PATH[split]
    with open(data_path) as f:
        d = json.load(f)

    source_contexts = {}
    if source_context_override_path is not None:
        with open(source_context_override_path) as f:
            source_contexts = json.load(f)

        # These aren't output in a pre-tokenized format like the
        # original source contexts, so we take care of that here.
        source_contexts_tok = {}
        for k, v in source_contexts.items():
            source_contexts_tok[k] = [tok.text for s in v for tok in nlp(s)]

    for ex in d:
        if ex["instance_id"] in source_contexts:
            # override source text
            ex["source"] = source_contexts_tok[ex["instance_id"]]
        yield ex


# Caches the original FAMuSSUM splits (no source text overrides)
SEAMUS_TRAIN = Dataset.from_generator(partial(gen, split="train"))
SEAMUS_DEV = Dataset.from_generator(partial(gen, split="dev"))
SEAMUS_TEST = Dataset.from_generator(partial(gen, split="test"))

if __name__ == "__main__":
    for ex in gen("test"):
        pass
