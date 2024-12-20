import json
import os
import spacy

from datasets import Dataset
from copy import deepcopy
from functools import partial
from glob import glob
from tqdm import tqdm
from typing import Any, Dict, Iterable, Iterator, Optional

from seamus.constants import SPLIT_TO_PATH, PARAPHRASES_SPLIT_TO_PATH, PARAPHRASE_TYPES

nlp = spacy.load("en_core_web_sm")


def load_paraphrases(
    split: str, paraphrase_types: Iterable[str] = set()
) -> Iterator[Dict[str, Any]]:
    """Load SEAMuS examples using paraphrased versions of source documents

    :param split: the split for which to generate paraphrased examples
    :param paraphrase_types: the paraphrase types for which to generate examples
    :return: SEAMuS examples with paraphrased source documents
    """

    def seamus_key_to_megawika_key(
        seamus_key: str, paraphrase_type: Optional[str] = None
    ) -> str:
        megawika_key = "-".join(seamus_key.split("-")[:3]).lower()
        if paraphrase_type is not None:
            megawika_key = megawika_key + "-" + paraphrase_type
        return megawika_key

    if split == "train":
        raise ValueError("No paraphrases available for train split.")

    # Load original SEAMuS data first
    with open(SPLIT_TO_PATH[split]) as f:
        d = json.load(f)

    # Are we using ALL kinds of paraphrases (blog, book, news, radio, reddit)
    # or only a subset?
    if not paraphrase_types:
        paraphrase_types = PARAPHRASE_TYPES

    # Load paraphrased source documents
    p = {}
    for paraphrase_file in glob(
        os.path.join(PARAPHRASES_SPLIT_TO_PATH[split], "*.jsonl")
    ):
        paraphrase_type = os.path.basename(paraphrase_file).split("_")[0]
        if paraphrase_type not in paraphrase_types:
            continue
        with open(paraphrase_file) as f:
            for line in tqdm(f, desc=f"Loading examples from {paraphrase_file}"):
                ex = json.loads(line)
                p[ex["id"]] = [tok.text for tok in nlp(ex["contents"])]

    # Create one new example per source document paraphrase
    # WARNING: even though we use the paraphrases in place of the
    #          original source documents, all other attributes remain
    #          unchanged, meaning that the source template annotations
    #          are no longer valid. You should not be using the source
    #          templates anyway if you are working with paraphrased
    #          source texts.
    for ex in d:
        ex["instance_id"] = seamus_key_to_megawika_key(ex["instance_id"])
        for paraphrase_type in sorted(paraphrase_types):
            p_ex = deepcopy(ex)
            p_ex["instance_id"] = (
                ex["instance_id"] + "-" + paraphrase_type
            )  # paraphrase example ID
            p_ex["source"] = p[p_ex["instance_id"]]  # paraphrased source
            yield p_ex


def gen(
    split: str,
    split_override: Optional[str] = None,
    source_context_override_path: Optional[str] = None,
    include_paraphrases: bool = False,
    paraphrase_types: Iterable[str] = set(),
) -> Iterator[Dict[str, Any]]:
    """Generate examples from the SEAMuS dataset

    :param split: The split to generate examples from
    :param split_override: Path to a JSON file containing examples to use
        in lieu of the original split.
    :param source_context_override_path: Path to a JSON file containing text
        to use in place of the original source document. For experiments, we
        want to use smaller selections from the source documents obtained via
        retrieval methods (see seamus/retrieval/)
    :param include_paraphrases: Whether to include GPT-generated paraphrases
        of the original SEAMuS source documents as additional training data
        (each original SEAMuS report is paired with *all* paraphrases)
    :param paraphrase_types: If include_paraphrases=True, which paraphrase
        types to include (options: blog, book, news, radio, reddit). If unset,
        defaults to all.
    """
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

    for ex in tqdm(d, desc="Loading original SEAMuS examples..."):
        if ex["instance_id"] in source_contexts:
            # override source text
            ex["source"] = source_contexts_tok[ex["instance_id"]]
        yield ex

    if include_paraphrases:
        yield from load_paraphrases(split, paraphrase_types=paraphrase_types)


# Caches the original FAMuSSUM splits (no source text overrides)
SEAMUS_TRAIN = Dataset.from_generator(partial(gen, split="train"))
SEAMUS_DEV = Dataset.from_generator(partial(gen, split="dev"))
SEAMUS_TEST = Dataset.from_generator(partial(gen, split="test"))

if __name__ == "__main__":
    tot = 0
    for ex in gen("dev", include_paraphrases=False, paraphrase_types={"news"}):
        tot += 1
    print(tot)
