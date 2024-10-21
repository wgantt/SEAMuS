import click
import evaluate
import json
import numpy as np


from seamus.constants import TEST_PATH, DETOKENIZER

ROUGE = evaluate.load("rouge")
METEOR = evaluate.load("meteor")
BERT_SCORE = evaluate.load("bertscore")


@click.command()
@click.argument("predictions", type=str)
@click.argument("output_file", type=str)
@click.option(
    "--task",
    "-t",
    type=click.Choice(["report-only", "combined"]),
    default="report-only",
)
def score(predictions, output_file, task) -> None:
    with open(predictions, "r") as f:
        predictions = [json.loads(line) for line in f]

    with open(TEST_PATH, "r") as f:
        references = json.load(f)

    predicted_summaries = []
    reference_summaries = []
    key = "report_summary" if task == "report-only" else "combined_summary"
    for p, r in zip(predictions, references):
        assert p["instance_id"] == r["instance_id"], "Mismatched instance IDs"
        predicted_summaries.append(p["response"])
        reference_summaries.append(DETOKENIZER.detokenize(r[key]))

    result = ROUGE.compute(
        predictions=predicted_summaries,
        references=reference_summaries,
        use_stemmer=True,
    )
    result["meteor"] = METEOR.compute(
        predictions=predicted_summaries, references=reference_summaries
    )["meteor"]
    bertscore = BERT_SCORE.compute(
        predictions=predicted_summaries, references=reference_summaries, lang="en"
    )
    for m, metric in zip(["p", "r", "f1"], ["precision", "recall", "f1"]):
        result["bertscore_" + m] = np.mean(bertscore[metric])

    out = {k: round(v, 4) for k, v in result.items()}
    with open(output_file, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    score()
