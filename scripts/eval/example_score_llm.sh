#!/bin/bash
# RUN ME FROM THE PROJECT ROOT!
SCORE=seamus.evaluation.score_llm
TASK="combined"

EXAMPLE_PRED_FILE="model_outputs/combined/text_with_event/claude-3-5-sonnet-20240620-0-shot.jsonl"
EXAMPLE_OUTPUT_FILE="score_llm_out.json"

python -m $SCORE $EXAMPLE_PRED_FILE $EXAMPLE_OUTPUT_FILE --task $TASK