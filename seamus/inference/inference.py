import click
import evaluate
import json
import logging
import numpy as np
import os
import sys
import torch

from datasets import Dataset
from functools import partial
from more_itertools import batched
from pprint import pprint
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.pipelines.pt_utils import KeyDataset
from typing import Dict, List

from seamus.constants import DETOKENIZER
from seamus.datasets.dataset import SEAMUS_TEST, gen
from seamus.training.train import preprocess, MAX_SUMMARY_LENGTH

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

ROUGE = evaluate.load("rouge")
METEOR = evaluate.load("meteor")
BERT_SCORE = evaluate.load("bertscore")


@click.command()
@click.argument("model_path", type=str)
@click.argument("hub_name", type=str)
@click.option(
    "--data-path",
    type=str,
    default=None,
    help="Path to data on which to run inference (if None, will load SEAMuS test split)",
)
@click.option(
    "--source-override-path",
    type=str,
    default=None,
)
@click.option(
    "--task",
    type=click.Choice(["report-only", "combined"]),
    default="report-only",
    help="the task to evaluate",
)
@click.option(
    "--device",
    type=click.INT,
    default=0,
    help="the device to on which to load the model",
)
@click.option("--batch-size", type=click.INT, default=2, help="the batch size to use")
@click.option(
    "--num-beams",
    type=click.INT,
    default=5,
    help="number of beams to use for beam search decoding",
)
@click.option(
    "--num-return-sequences",
    type=click.INT,
    default=1,
    help="number of summaries to generate",
)
@click.option(
    "--max-doc-len",
    type=click.INT,
    default=None,
    help="maximum number of tokens in the input document (all longer docs will be truncated)",
)
@click.option(
    "--min-new-tokens",
    type=click.INT,
    default=15,
    help="the minimum number of tokens to generate in the summary (for evaluation)",
)
@click.option(
    "--max-new-tokens",
    type=click.INT,
    default=MAX_SUMMARY_LENGTH,
    help="maximum number of tokens to generate in the summary (for evaluation)",
)
@click.option(
    "--num-beams",
    type=click.INT,
    default=5,
    help="number of beams to use for beam search (eval loop only)",
)
@click.option(
    "--input-format",
    type=click.Choice(
        [
            "event_only",
            "text_only",
            "text_with_schema",
            "text_with_event",
        ]
    ),
    default="text_with_event",
    help="the input format",
)
def inference(
    model_path,
    hub_name,
    data_path,
    source_override_path,
    task,
    device,
    batch_size,
    num_beams,
    min_new_tokens,
    max_new_tokens,
    num_return_sequences,
    max_doc_len,
    input_format,
) -> None:
    """Run inference for FAMuSSUM models

    :param model_path: path to the model with which to run inference
    :param hub_name: the name of the underlying model on HuggingFace Hub
    :param data_path: path to data on which to run inference (if None, will load FAMuSSUM test split)
    :param source_override_path: path to the source text override
    :param device: The GPU on which the model will be loaded
    :param batch_size: The batch size to use
    :param num_beams: The number of beams to use for beam seearch
    :param min_new_tokens: minimum number of tokens to generate in the summary
    :param max_new_tokens: maximum number of tokens to generate in the summary
    :param max_doc_len: maximum number of tokens in the input document (all longer docs will be truncated)
    :param input_format: how the input should be formatted
    :param num_return_sequences: The number of summaries to generate for each
        example
    :returns None:
    """
    logger.warning(f"Loading model {model_path} for inference...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    logger.warning("Done!")
    if "t5" in hub_name:
        # default behavior of `from_pretrained` here is apparently incorrect for T5; see below:
        if hub_name in {"t5-small", "t5-base"}:
            model_max_length = 512
        else:
            model_max_length = 1024
        tokenizer = AutoTokenizer.from_pretrained(
            hub_name, model_max_length=model_max_length
        )
        # required by T5
        prefix = ["summarize", ":"]
    elif "bart" in hub_name:
        tokenizer = AutoTokenizer.from_pretrained(hub_name, add_prefix_space=True)
        prefix = []
    else:
        tokenizer = AutoTokenizer.from_pretrained(hub_name)
        prefix = []

    preprocess_fn = partial(
        preprocess,
        # model=hub_name,
        tokenizer=tokenizer,
        max_doc_len=max_doc_len,
        input_format=input_format,
        prefix=prefix,
        task=task,
    )

    if data_path is None:
        test_data = SEAMUS_TEST
        output_file_prefix = "test"
    else:
        test_data = Dataset.from_generator(
            partial(
                gen,
                split="test",
                split_override=data_path,
                source_context_override_path=source_override_path,
            )
        )
        output_file_prefix = os.path.basename(data_path)[:-5]  # drop '.json'

    test_data = test_data.map(
        preprocess_fn,
        batched=True,
    )
    preds = []
    summary_key = "report_summary" if task == "report-only" else "combined_summary"
    refs = [DETOKENIZER.detokenize(ex[summary_key]) for ex in test_data]
    expected_iters = len(test_data) // batch_size
    for batch_idxs in tqdm(
        batched(range(len(test_data)), batch_size),
        desc="Evaluating...",
        total=expected_iters,
    ):
        batch = test_data.select(batch_idxs)
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )
        preds.extend(
            [
                summary.strip()
                for summary in tokenizer.batch_decode(output, skip_special_tokens=True)
            ]
        )

    scores_dict = score(preds, refs)
    pprint(scores_dict)

    predictions = []
    for ex, pred, ref in zip(test_data, preds, refs):
        # TODO: add reference
        predictions.append(
            {
                "instance_id": ex["instance_id"],
                "format": input_format,
                "task": task,
                "prediction": pred,
                "reference": ref,
            }
        )

    predictions_file = os.path.join(model_path, f"../{output_file_prefix}_preds.jsonl")
    with open(predictions_file, "w") as f:
        f.write("\n".join(list(map(json.dumps, predictions))))

    with open(
        os.path.join(model_path, f"../{output_file_prefix}_preds_pretty.json"), "w"
    ) as f:
        json.dump(predictions, f, indent=2)

    scores_file = os.path.join(model_path, f"../{output_file_prefix}_metrics.json")
    with open(scores_file, "w") as f:
        json.dump(scores_dict, f, indent=2)


def score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE metrics for generated summaries

    :param predictions: the predicted summaries
    :param references: the reference summaries
    :returns: a dictionary of rouge-1, rouge-2, and rouge-L scores
    """
    result = ROUGE.compute(
        predictions=predictions, references=references, use_stemmer=True
    )
    bertscore = BERT_SCORE.compute(
        predictions=predictions, references=references, lang="en"
    )
    result["meteor"] = METEOR.compute(predictions=predictions, references=references)[
        "meteor"
    ]
    for m, metric in zip(["p", "r", "f1"], ["precision", "recall", "f1"]):
        result["bertscore_" + m] = np.mean(bertscore[metric])

    return {k: round(v, 4) for k, v in result.items()}


if __name__ == "__main__":
    inference()
