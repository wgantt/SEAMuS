#!/bin/bash
# RUN ME FROM THE PROJECT ROOT!
SCORE=seamus.evaluation.ceaf_ree
PREDICTIONS_ROOT="model_outputs"
TASK="combined" # or report_only
METRIC="subset" # or edit_distance
SPLIT="test" # should match the split for which model predictions are being evaluated

EXAMPLE_PRED_FILE="model_outputs/combined/text_with_event/spanfinder_out/t5-bm25_concat_7-1337_spanfinder.jsonl"
EXAMPLE_OUTPUT_FILE="ceaf_ree_example_output.json"

python -m $SCORE $EXAMPLE_PRED_FILE --split $SPLIT --metric $METRIC --task $TASK --output-file $EXAMPLE_OUTPUT_FILE