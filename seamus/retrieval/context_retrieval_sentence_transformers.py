import click
import json
import os
import torch

from sentence_transformers import SentenceTransformer
from typing import Dict, List

from seamus.constants import SPLIT_TO_PATH, SAVED_CONTEXTS_PATH, DETOKENIZER
from seamus.retrieval.utils import detokenize_text, sentence_split_text
from tqdm import tqdm

# Models we tried:
# - all-mpnet-base-v2
# - embaas/sentence-transformers-e5-large-v2


def retrieve_source_context_transformers(
    model_name: str,
    split: str,
    context_window_size: int = 5,
    mode: str = "concat",
    **transformer_kwargs,
) -> Dict[str, List[str]]:
    """Select a context window from the source for a given report using a sentence transformer model

    :param model_name: The name of the sentence transformer model to use
    :param split: The split to retrieve text from
    :param context_window_size: Number of sentences to add as context. For
        "expand" mode, this is the number of sentences to include on either
        side of the best retrieved sentence. For "concat" mode, this is the
        k value in the top-k sentences to retrieve.
    :param mode: The method to use for selecting context. Options are "expand"
        and "concatenate" (see CONTEXT_SELECTION_MODES at top of file)
    :param transformer_kwargs: Additional keyword arguments to pass to the SentenceTransformer
    :return: A dictionary mapping instance IDs to a list of context sentences
    """

    # load data
    with open(SPLIT_TO_PATH[split], "r") as f:
        data = json.load(f)
        data = {ex["instance_id"]: ex for ex in data}

    # total source arguments that appear within the context window
    source_args_in_context = 0

    # total arguments across all source templates
    total_source_args = 0

    # load the sentence transformer model
    model = SentenceTransformer(model_name, **transformer_kwargs)

    # report and source text comes whitespace tokenized;
    # we detokenize first for input to the sentence transformer
    report_texts = detokenize_text(split, "report")
    source_sents = sentence_split_text(split, "source")
    contexts = {}
    for example_id, report in tqdm(report_texts.items(), desc="Retrieving contexts..."):
        source = source_sents[example_id]

        # The query is the entire report text
        query_embedding = model.encode([report])

        # The corpus is the sentence-split source text
        doc_embeddings = model.encode(source)

        # Compute cosine similarity between the query and each source sentence
        scores = torch.tensor(query_embedding @ doc_embeddings.T)

        if mode == "expand":
            # In 'expand' mode, we select the best sentence and include
            # the k sentences to the left and the k sentences to the right
            best_score, best_idx = scores.topk(1)
            best_sentence_idx = best_idx.item()
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
            best_scores, best_idxs = scores.topk(k)
            # Note: we sort here to ensure sentences are in document order
            contexts[example_id] = [source[i] for i in sorted(best_idxs[0])]
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
@click.option("--model-name", "-n", type=str, default="all-mpnet-base-v2")
@click.option("--output_path", "-o", type=str, default=None)
@click.option("--context-window-size", "-w", type=int, default=5)
@click.option("--mode", "-m", type=str, default="concat")
def get_source_contexts(split, output_path, model_name, context_window_size, mode):
    contexts = retrieve_source_context_transformers(
        model_name, split, context_window_size, mode
    )

    if output_path is None:
        model_name = model_name.split("/")[-1]
        output_path = f"{model_name}_{split}_{mode}_{context_window_size}.json"
        output_path = os.path.join(SAVED_CONTEXTS_PATH, output_path)

    with open(output_path, "w") as f:
        json.dump(contexts, f, indent=2)


if __name__ == "__main__":
    get_source_contexts()
