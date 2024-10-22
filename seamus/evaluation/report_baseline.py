import click
import evaluate
import json
import numpy as np

from alignscore import AlignScore
from pprint import pprint
from typing import Dict, List

from seamus.constants import TEST_PATH

# Basic summarization metrics
ROUGE = evaluate.load("rouge")
BERTSCORE = evaluate.load("bertscore")

# NOTE: You must download an AlignScore checkpoint to use this script
# See documentation here: https://github.com/yuh-zha/AlignScore
ALIGNSCORES_EVALUATION_MODES = frozenset({"nli_sp", "nli", "bin_sp", "bin"})
ALIGNSCORE_MODEL_CHECKPOINT = "..."  # SET ME!


@click.command()
@click.option(
    "--do-report-only",
    is_flag=True,
    default=False,
    help="Use the report summary as the reference (rather than the combined summary)",
)
@click.option(
    "--use-report-summary",
    is_flag=True,
    default=False,
    help="Use the report summary as the predicted summary (rather than the report text)",
)
def compute_report_baseline(do_report_only: bool, use_report_summary: bool):
    with open(TEST_PATH) as f:
        d = json.load(f)

    assert not (do_report_only and use_report_summary), "Cannot use both flags!"
    predictions = []
    references = []
    for ex in d:
        if use_report_summary:
            predictions.append(" ".join(ex["report_summary"]))
        else:
            predictions.append(" ".join(ex["report"]))

        if do_report_only:
            references.append(" ".join(ex["report_summary"]))
        else:
            references.append(" ".join(ex["combined_summary"]))

    results = {}
    results |= rouge(predictions, references)
    results |= bertscore(predictions, references)
    # results |= alignscore(predictions, references)
    pprint(results)


def rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE metrics for generated summaries

    :param predictions: the predicted summaries
    :param references: the reference summaries
    :returns: a dictionary of rouge-1, rouge-2, and rouge-L scores
    """
    result = ROUGE.compute(
        predictions=predictions, references=references, use_stemmer=True
    )
    return {k: round(v, 4) for k, v in result.items()}


def bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BERTScore metrics for generated summaries

    :param predictions: the predicted summaries
    :param references: the reference summaries
    :param model: the model to use for computing BERTScore
    :returns: a dictionary of P, R, F1 scores
    """
    out = BERTSCORE.compute(predictions=predictions, references=references, lang="en")
    result = {}
    for m, metric in zip(["p", "r", "f1"], ["precision", "recall", "f1"]):
        result["bertscore_" + m] = np.mean(out[metric])
    return result


def alignscore(
    predictions: List[str],
    references: List[str],
    evaluation_mode: str = "nli_sp",
    device: str = "cuda:0",
) -> Dict[str, float]:
    """Compute AlignScore metrics for generated summaries

    :param predictions: the predicted summaries
    :param references: the reference summaries
    :param model: the model to use for computing BERTScore
    :returns: a dictionary of AlignScore scores
    """
    assert (
        evaluation_mode in ALIGNSCORES_EVALUATION_MODES
    ), f"Invalid AlignScore evaluation mode: {evaluation_mode}!"
    scorer = AlignScore(
        model="roberta-large",
        batch_size=16,
        evaluation_mode=evaluation_mode,
        device=device,
        ckpt_path=ALIGNSCORE_MODEL_CHECKPOINT,
    )
    result = {
        "alignscore": np.mean(scorer.score(contexts=references, claims=predictions))
    }
    return result


if __name__ == "__main__":
    compute_report_baseline()
