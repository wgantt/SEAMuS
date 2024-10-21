#!/bin/bash
# RUN ME FROM THE PROJECT ROOT
PROMPT=seamus.prompt.prompt_gpt
MODEL=gpt-4o-mini-2024-07-18

# These are already the script defaults;
# do not change if you are interested in reproducing our results
MAX_TOKENS=256
TEMPERATURE=0.7

EXAMPLE_INPUT_FILE="seamus/prompt/test_prompt_file.jsonl"
EXAMPLE_OUTPUT_FILE="gpt_example_output.jsonl"

# SET ME!
export OPENAI_ORG_ID="..."
export OPENAI_API_KEY="..."

python -m $PROMPT $MODEL $EXAMPLE_INPUT_FILE $EXAMPLE_OUTPUT_FILE --max-tokens $MAX_TOKENS --temperature $TEMPERATURE