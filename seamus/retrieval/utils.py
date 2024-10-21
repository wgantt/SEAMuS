import json
import Stemmer

from spacy.lang.en import English
from typing import Dict, List

from seamus.constants import DETOKENIZER, SPLIT_TO_PATH, SPLITS, TEXT_FIELDS

# Expand: identify the best sentence and include all surrounding sentences
#         within some context window
# Concat: identify the k-best sentences and concatenate them together
CONTEXT_SELECTION_MODES = frozenset({"concat", "expand"})


stemmer = Stemmer.Stemmer("english")

nlp = English()
nlp.add_pipe("sentencizer")


def detokenize_text(split: str, field: str) -> Dict[str, str]:
    """Detokenize a tokenized text field

    :param split: The split to retrieve the text from
    :param field: The field to retrieve the text from
    :return: A dictionary mapping instance IDs to detokenized text for the given field
    """
    assert split in SPLITS, f"Invalid split: {split}"
    assert field in TEXT_FIELDS, f"Invalid text field: {field}"
    texts = {}
    with open(SPLIT_TO_PATH[split], "r") as f:
        data = json.load(f)
        for item in data:
            texts[item["instance_id"]] = DETOKENIZER.detokenize(item[field])
    return texts


def sentence_split_text(split: str, field: str) -> Dict[str, List[str]]:
    """Split a tokenized text field into sentences

    :param split: The split to retrieve the text from
    :param field: The field to retrieve the text from
    :return: A dictionary mapping instance IDs to a list of sentences for the given field
    """
    texts = detokenize_text(split, field)
    for example_id, text in texts.items():
        texts[example_id] = [sent.text for sent in nlp(text).sents]
    return texts
