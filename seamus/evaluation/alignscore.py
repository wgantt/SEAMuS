import os
import json
import argparse
import random
import jsonlines
import numpy as np

from alignscore import AlignScore
from tqdm import tqdm

from seamus.constants import TEST_PATH
from seamus.datasets.dataset import SEAMUS_TEST


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gen_json_path",
        type=str,
    )
    parser.add_argument(
        "ckpt_path", type=str
    )
    parser.add_argument("--data_path", type=str, default=TEST_PATH)
    parser.add_argument(
        "--setting",
        type=str,
        default="report_only",
        choices=["report_only", "combined"],
    )
    parser.add_argument("--model", type=str, default="roberta-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--evaluation_mode", type=str, default="nli_sp")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()
    return args


def bootstrap(scorer, gen_json, test_data, n_samples, seed, setting="report_only"):
    random.seed(seed)
    actual_scores = evaluate_model(scorer, gen_json, test_data, setting=setting)
    out_scores = {}
    sampled_scores = []
    for _ in tqdm(range(n_samples), desc="Sampling..."):
        scores = random.choices(actual_scores, k=len(actual_scores))
        sampled_scores.append(np.mean(scores))

    mean = np.round(np.mean(sampled_scores), 4)
    sd = np.round(np.std(sampled_scores), 4)
    ci_lo = np.round(np.percentile(sampled_scores, 2.5), 4)
    ci_hi = np.round(np.percentile(sampled_scores, 97.5), 4)
    out_scores[f"alignscore_mean"] = mean
    out_scores[f"alignscore_sd"] = sd
    out_scores[f"alignscore_2.5"] = ci_lo
    out_scores[f"alignscore_97.5"] = ci_hi
    print(f"Mean: {np.round(mean * 100, 2)}")
    print(f"Standard deviation: {np.round(sd * 100, 2)}")
    print(f"95% CI: {np.round(ci_lo * 100, 2)} - {np.round(ci_hi * 100, 2)}")
    print()
    return out_scores


def evaluate_model(scorer, gen_json, test_data, setting="report_only"):
    scores = []
    contexts = []
    claims = []
    for instance in test_data:
        instance_id = instance["instance_id"]
        report = instance["report"]
        report = " ".join(report)
        source = instance["source"]
        source = " ".join(source)

        if instance_id in gen_json:
            pred = gen_json[instance_id]["pred"]
            if setting == "report_only":
                contexts.append(report)
            elif setting == "combined":
                contexts.append(report + " " + source)
            claims.append(pred)
    return scorer.score(contexts=contexts, claims=claims)


def main():
    args = parse_args()
    scorer = AlignScore(
        model=args.model,
        batch_size=args.batch_size,
        device=args.device,
        ckpt_path=args.ckpt_path,
        evaluation_mode=args.evaluation_mode,
    )

    test_data = SEAMUS_TEST
    generated_data = []
    with jsonlines.open(args.gen_json_path) as f:
        for line in f:
            generated_data.append(line)

    generated_data_json = {}
    for example in generated_data:
        instance_id = example["instance_id"]
        if "response" in example:
            pred = example["response"]
        elif "prediction" in example:
            pred = example["prediction"]
        else:
            raise ValueError("No response or prediction key found in generated output")

        generated_data_json[instance_id] = {
            "pred": pred,
        }

    if args.bootstrap:
        scores_dict = bootstrap(
            scorer,
            generated_data_json,
            test_data,
            n_samples=args.n_samples,
            seed=args.seed,
            setting=args.setting,
        )
        if args.output_file is None:
            output_file = (
                os.path.dirname(args.gen_json_path)
                + "/"
                + os.path.basename(args.gen_json_path).replace(
                    ".jsonl", "_bootstrap_alignscore.json"
                )
            )
            with open(output_file, "w") as f:
                json.dump(scores_dict, f, indent=2)
    else:
        scores = evaluate_model(
            scorer, generated_data_json, test_data, setting=args.setting
        )


if __name__ == "__main__":
    main()
