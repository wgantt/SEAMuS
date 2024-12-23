import click
import bm25s
import json
import os

from typing import Dict, List, Optional

from seamus.constants import (
    DETOKENIZER,
    SPLIT_TO_PATH,
    SAVED_CONTEXTS_SPLIT_TO_PATH,
    PARAPHRASES_SPLIT_TO_PATH,
    PARAPHRASE_TYPES,
    seamus_key_to_megawika_key,
    ID_TYPES,
)
from seamus.retrieval.utils import (
    detokenize_text,
    sentence_split_text,
    stemmer,
)
from tqdm import tqdm


def retrieve_source_context_bm25(
    split: str,
    context_window_size: int = 2,
    mode: str = "concat",
    paraphrase_type: Optional[str] = None,
    id_type: str = "seamus",
) -> Dict[str, List[str]]:
    """Select a context window from the source for a given report using BM25

    :param split: The split to retrieve text from
    :param context_window_size: Number of sentences to add as context. For
        "expand" mode, this is the number of sentences to include on either
        side of the best retrieved sentence. For "concat" mode, this is the
        k value in the top-k sentences to retrieve.
    :param mode: The method to use for selecting context. Options are "expand"
        and "concatenate" (see CONTEXT_SELECTION_MODES at top of file)
    :param paraphrase_type: if non-None, will retrieve sentences using the
        corresponding paraphrased version of the source context, rather than
        the original source text.
    :param id_type: whether to use SEAMuS- or MegaWika-style example IDs
        in the output
    :return: A dictionary mapping instance IDs to a list of context sentences
    """
    with open(SPLIT_TO_PATH[split], "r") as f:
        seamus_data = json.load(f)
        seamus_data = {ex["instance_id"]: ex for ex in seamus_data}

    # load data
    if paraphrase_type is not None:
        assert (
            paraphrase_type in PARAPHRASE_TYPES
        ), f"Invalid paraphrase type '{paraphrase_type}'. Choices are: {', '.join(PARAPHRASE_TYPES)}"
        with open(
            os.path.join(
                PARAPHRASES_SPLIT_TO_PATH[split], f"{paraphrase_type}_{split}.jsonl"
            ),
            "r",
        ) as f:
            data = [json.loads(line) for line in f]
            data = {ex["id"]: ex for ex in data}
    else:
        data = seamus_data

    # total source arguments that appear within the context window
    source_args_in_context = 0

    total_source_args = 0

    bm25 = bm25s.BM25()

    # report and source text comes whitespace tokenized;
    # must detokenize to work with bm25s
    report_texts = detokenize_text(split, "report")
    source_sents = sentence_split_text(split, "source", paraphrase_type)
    contexts = {}
    for seamus_example_id, report in tqdm(
        report_texts.items(), desc="Retrieving contexts..."
    ):
        if id_type == "megawika":
            example_id = seamus_key_to_megawika_key(seamus_example_id, paraphrase_type)
        else:
            example_id = seamus_example_id

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
        ex = seamus_data[seamus_example_id]
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
@click.option("--paraphrase-type", "-p", type=click.Choice(PARAPHRASE_TYPES))
@click.option("--id-type", "-t", type=click.Choice(ID_TYPES), default="seamus")
def get_source_contexts(
    split, output_path, context_window_size, mode, paraphrase_type, id_type
):
    contexts = retrieve_source_context_bm25(
        split, context_window_size, mode, paraphrase_type, id_type
    )

    if output_path is None:
        if paraphrase_type is None:
            output_path = f"bm25_{split}_{mode}_{context_window_size}.json"
        else:
            output_path = (
                f"bm25_{paraphrase_type}_{split}_{mode}_{context_window_size}.json"
            )
        output_path = os.path.join(SAVED_CONTEXTS_SPLIT_TO_PATH[split], output_path)

    with open(output_path, "w") as f:
        json.dump(contexts, f, indent=2)


if __name__ == "__main__":
    get_source_contexts()
