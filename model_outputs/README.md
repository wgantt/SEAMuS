# Model Outputs

This directory contains the predictions and the metric scores for all results reported in the paper, with the exception of FActScore, which we are working on adding. `report_only.zip` contains results for the SEAMuS report summarization task and `combined.zip` contains results for the combined summarization task.

Within each of these archives, you will find five subdirectories:
- `corrupted`: results of the annotation corruption experiments reported in Section 4.4 and in Table 3 of the paper.
- `event_only`: results of the Event Only ablation, reported in Tables 6 and 7 in the appendix.
- `text_only`: results of the Text Only ablation, reported in Tables 6 and 7 in the appendix.
- `text_with_schema`: results of the Text+Schema ablation, reported in Tables 6 and 7 in the appendix.
- `text_with_event`: results of the *unablated* setting, reported in Table 2 and (redundantly) in the Text+Event lines in Tables 6 and 7.

There are also a couple of files for the *report baseline* in each archive:
- `report_baseline.json`: contains metric scores for the report baseline.
- `report_baseline_spanfinder_out.jsonl`: contains predicted arguments from each report text, used to compute CEAF-REE (this file is the same in each archive).

Some details on the file naming conventions:
- Every file name starts with a detailed model specifier. For the Claude and GPT models, the names also include `0-shot` or `few-shot`, depending on the setting in which they were evaluated.
- For files in the `corrupted/` subdirectories, the model names include a string indicating the value of `p` used to obtain those results (e.g. `0.20`).
- All `.jsonl` files contain model predictions in JSON lines format. For the fine-tuned models, the predictions are given in the `prediction` field. For the Claude and GPT models, the predictions are given in the `response` field.
- All files ending in `scores.json` contain metric results for ROUGE-1, ROUGE-2, ROUGE-LCS, BERTScore, and CEAF-REE (both the original `subset` version and the soft-match `edit_distance` version).
- All files ending in `bootstrap.json` contain summary statistics for these same metrics based on a non-parametric bootstrap (with n=1,000)
- All files ending in `bootstrap_alignscore.json` contain summary statistics for AlignScore &mdash; also based on a non-parametric bootstrap (with n=1,000)

Lastly, the files in `spanfinder_in` and `spanfinder_out` can be ignored, unless you want to compute the CEAF-REE metric yourself. We will add instructions for how to do this soon.