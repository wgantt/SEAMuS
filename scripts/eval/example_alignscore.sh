#!/bin/bash
# RUN ME FROM THE PROJECT ROOT!
ALIGNSCORE="seamus.evaluation.alignscore"
TASK="combined"

# Don't change these if you're trying to replicate our results
SEED=1337
N_SAMPLES=1000

# Change these as needed
EXAMPLE_PRED_FILE="model_outputs/combined/text_with_event/t5-bm25_concat_7-1337.jsonl"
LOG=alignscore.log

python -m $ALIGNSCORE $EXAMPLE_PRED_FILE --setting $TASK --n-samples $N_SAMPLES --seed $SEED --bootstrap | tee -a $LOG