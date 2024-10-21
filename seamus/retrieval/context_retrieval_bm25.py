import click
import bm25s
import json
import os

from typing import Dict, List

from seamus.constants import DETOKENIZER, SPLIT_TO_PATH, SAVED_CONTEXTS_PATH
from seamus.retrieval.utils import (
    detokenize_text,
    sentence_split_text,
    stemmer,
)
from tqdm import tqdm


def retrieve_source_context_bm25(
    split: str, context_window_size: int = 2, mode: str = "concat"
) -> Dict[str, List[str]]:
    """Select a context window from the source for a given report using BM25

    :param split: The split to retrieve text from
    :param context_window_size: Number of sentences to add as context. For
        "expand" mode, this is the number of sentences to include on either
        side of the best retrieved sentence. For "concat" mode, this is the
        k value in the top-k sentences to retrieve.
    :param mode: The method to use for selecting context. Options are "expand"
        and "concatenate" (see CONTEXT_SELECTION_MODES at top of file)
    :return: A dictionary mapping instance IDs to a list of context sentences
    """

    # load data
    with open(SPLIT_TO_PATH[split], "r") as f:
        data = json.load(f)
        data = {ex["instance_id"]: ex for ex in data}

    # total source arguments that appear within the context window
    source_args_in_context = 0

    total_source_args = 0

    bm25 = bm25s.BM25()

    # report and source text comes whitespace tokenized;
    # must detokenize to work with bm25s
    report_texts = detokenize_text(split, "report")
    source_sents = sentence_split_text(split, "source")
    contexts = {}
    for example_id, report in tqdm(report_texts.items(), desc="Retrieving contexts..."):
        source = source_sents[example_id]

        # The corpus is just the set of source sentences for this example
        corpus_tokens = bm25s.tokenize(source, stopwords="en", stemmer=stemmer)
        bm25.index(corpus_tokens)

        # The query is the entire report text
        query = report
        query_tokens = bm25s.tokenize(query, stemmer=stemmer)

        if mode == "expand":
            # In 'expand' mode, we select the best sentence and include
            # the k sentences to the left and the k sentences to the right
            results, scores = bm25.retrieve(query_tokens, k=1)
            best_sentence_idx = results[0, 0]
            if best_sentence_idx - context_window_size < 0:
                start_idx = 0
                end_idx = 4  # inclusive
            else:
                start_idx = best_sentence_idx - context_window_size
                end_idx = best_sentence_idx + context_window_size  # inclusive
            contexts[example_id] = source[start_idx : end_idx + 1]
            context_text = " ".join(source[start_idx : end_idx + 1])
        elif mode == "concat":
            # In 'concatenate' mode, we select the k-best sentences
            # and just concatenate them
            k = min(context_window_size, len(source))
            results, scores = bm25.retrieve(query_tokens, k=k)
            # Note: we sort here to ensure sentences are in document order
            contexts[example_id] = [source[i] for i in sorted(results[0])]
            context_text = " ".join(contexts[example_id])
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # See how many source arguments actually
        # appear in the retrieved context
        ex = data[example_id]
        for role, role_data in ex["source_template"].items():
            for arg in role_data["arguments"]:
                total_source_args += 1
                arg_text = DETOKENIZER.detokenize(arg["tokens"])
                if arg_text in context_text:
                    source_args_in_context += 1

    # Show only train and dev statistics
    if split != "test":
        print(f"Total source arguments: {total_source_args}")
        print(f"Source arguments in context: {source_args_in_context}")
        print(
            f"Percentage of source arguments in context: {(source_args_in_context / total_source_args) * 100:.2f}%"
        )

    return contexts


@click.command()
@click.argument("split", type=str)
@click.option("--output_path", "-o", type=str, default=None)
@click.option("--context-window-size", "-w", type=int, default=5)
@click.option("--mode", "-m", type=str, default="concat")
def get_source_contexts(split, output_path, context_window_size, mode):
    contexts = retrieve_source_context_bm25(split, context_window_size, mode)

    if output_path is None:
        output_path = f"bm25_{split}_{mode}_{context_window_size}.json"
        output_path = os.path.join(SAVED_CONTEXTS_PATH, output_path)

    with open(output_path, "w") as f:
        json.dump(contexts, f, indent=2)


if __name__ == "__main__":
    get_source_contexts()
