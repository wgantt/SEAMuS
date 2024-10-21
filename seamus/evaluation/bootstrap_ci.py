import click
import evaluate
import json
import numpy as np
import os
import random

from collections import defaultdict
from tqdm import tqdm

from seamus.constants import SPLIT_TO_PATH, DETOKENIZER
from seamus.evaluation.ceaf_ree import (
    ceaf_ree_phi_subset_metrics,
    ceaf_ree_phi_edit_distance_metrics,
    extract_gold_event_structure,
    extract_predicted_event_structure,
)

ROUGE = evaluate.load("rouge")
METEOR = evaluate.load("meteor")
BERT_SCORE = evaluate.load("bertscore")
CEAF_REE_PHI_SUBSET = ceaf_ree_phi_subset_metrics.new()
CEAF_REE_PHI_EDIT_DISTANCE = ceaf_ree_phi_edit_distance_metrics.new()


TEST_PATH = SPLIT_TO_PATH["test"]


@click.command()
@click.argument("preds-file", type=str)
@click.argument("pred-spans-file", type=str)
@click.argument("task", type=click.Choice(["report_only", "combined"]))
@click.option("--output-file", "-o", type=str, default=None)
@click.option("--pred-key", "-k", type=str, default="response")
@click.option("--num-samples", "-n", type=int, default=100)
@click.option("--seed", "-s", type=int, default=1337)
def bootstrap(
    preds_file, pred_spans_file, task, output_file, pred_key, num_samples, seed
):
    random.seed(seed)
    with open(preds_file) as f:
        pred_summaries = [json.loads(line)[pred_key] for line in f]

    with open(pred_spans_file) as f:
        pred_spans = [json.loads(line) for line in f]

    ref_key = "report_summary" if task == "report_only" else "combined_summary"
    print(f"ref_key={ref_key}")
    with open(TEST_PATH) as f:
        test_data = json.load(f)
        ref_summaries = [DETOKENIZER.detokenize(d[ref_key]) for d in test_data]

    result = ROUGE.compute(
        predictions=pred_summaries,
        references=ref_summaries,
        use_stemmer=True,
        use_aggregator=False,
    )
    result["bertscore_f1"] = BERT_SCORE.compute(
        predictions=pred_summaries, references=ref_summaries, lang="en"
    )["f1"]
    pred_event_structures = [extract_predicted_event_structure(ex) for ex in pred_spans]
    ref_event_structures = [extract_gold_event_structure(ex, task) for ex in test_data]
    result["ceaf_ree_phi_subset_f1"] = []
    result["ceaf_ree_phi_edit_distance_f1"] = []
    for p, r in zip(pred_event_structures, ref_event_structures):
        CEAF_REE_PHI_SUBSET.reset()
        CEAF_REE_PHI_EDIT_DISTANCE.reset()
        CEAF_REE_PHI_SUBSET.update_single([p], [r])
        CEAF_REE_PHI_EDIT_DISTANCE.update_single([p], [r])
        result["ceaf_ree_phi_subset_f1"].append(CEAF_REE_PHI_SUBSET.compute()["f1"])
        result["ceaf_ree_phi_edit_distance_f1"].append(
            CEAF_REE_PHI_EDIT_DISTANCE.compute()["f1"]
        )

    pop = range(len(test_data))
    estimates = defaultdict(list)
    for _ in tqdm(range(num_samples), desc="Sampling..."):
        sampled_ids = random.choices(pop, k=len(test_data))
        for k, v in result.items():
            estimate = np.mean([v[idx] for idx in sampled_ids])
            estimates[k].append(estimate)

    out_scores = {}
    for metric, scores in estimates.items():
        scores = np.array(scores)
        mean = np.round(np.mean(scores), 4)
        sd = np.round(np.std(scores), 4)
        ci_lo = np.round(np.percentile(scores, 2.5), 4)
        ci_hi = np.round(np.percentile(scores, 97.5), 4)
        out_scores[f"{metric}_mean"] = mean
        out_scores[f"{metric}_sd"] = sd
        out_scores[f"{metric}_2.5"] = ci_lo
        out_scores[f"{metric}_97.5"] = ci_hi
        print(f"{metric}")
        print(f"Mean: {np.round(mean * 100, 2)}")
        print(f"Standard deviation: {np.round(sd * 100, 2)}")
        print(f"95% CI: {np.round(ci_lo * 100, 2)} - {np.round(ci_hi * 100, 2)}")
        print()

    if output_file is None:
        output_file = (
            os.path.dirname(preds_file)
            + "/"
            + os.path.basename(preds_file).replace(".jsonl", "_bootstrap.json")
        )
    with open(output_file, "w") as f:
        json.dump(out_scores, f, indent=2)


if __name__ == "__main__":
    bootstrap()
