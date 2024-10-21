#!/bin/bash
# RUN ME FROM THE PROJECT ROOT!
BOOSTRAP_CI="seamus.evaluation.bootstrap_ci"
MODEL_OUTPUTS_DIR="model_outputs/"
TASK="combined" # or report_only
PRED_KEY="prediction" # "prediction" for small models; "response" for Claude and GPT
SEED=1337 # don't change this if you're trying to replicate our results
N_SAMPLES=1000 # also don't change this if you're trying to replicate our results
LOG="bootstrap_ci_combined.log" # make the log file whatever you want

# Make these whatever you want
EXAMPLE_PRED_FILE="model_outputs/combined/text_with_event/t5-bm25_concat_7-1337.jsonl"
EXAMPLE_PRED_SPANS_FILE="model_outputs/combined/text_with_event/spanfinder_out/t5-bm25_concat_7-1337_spanfinder.jsonl"
EXAMPLE_OUTPUT_FILE="bootstrap.json"

python -m $BOOSTRAP_CI $EXAMPLE_PRED_FILE $EXAMPLE_PRED_SPANS_FILE $TASK --pred-key $PRED_KEY -n $N_SAMPLES --seed $SEED -o $EXAMPLE_OUTPUT_FILE | tee -a $LOG