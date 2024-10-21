# Evaluation

Contents:
- `alignscore.py`: Script for computing AlignScore results. Unfortunately, the dependencies required by the [AlignScore repo](https://github.com/yuh-zha/AlignScore) are incompatible with those required by the rest of this project. We are working on putting together separate instructions for creating an environment that will enable you to run this script.
- `bootstrap_ci.py`: Computes summary statistics (mean, standard deviation, 95% CIs) based on a nonparametric bootstrap of a model's outputs.
- `score_llm.py`: Computes ROUGE-1, ROUGE-2, ROUGE-LCS, and BERTScore results on LLM predictions (which can be found in the `model_outputs` directory)
- `ceaf_ree.py`: Computes both the strict version of CEAF-REE and the soft-match version reported in the paper.

A couple notes on script arguments:
- For `ceaf_ree.py`, the `pred_file` argument should be taken from one of the `spanfinder_out/` directories in `model_outputs/{report_only,combined}/{corrupted,event_only,text_only,text_with_schema,text_with_event}/`. These files (and no others) contain the argument spans extracted from the predicted summaries using the span extraction model discussed in Section 4.1.
- You should also use a `spanfinder_out/` file as the argument for the `pred-spans-file` argument for bootstrap\_ci.py. (The `preds-file` argument is an ordinary model outputs file.)

For examples of how to run these scripts, see `scripts/eval/`.