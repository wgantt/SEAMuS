#!/bin/bash

BUILD_PROMPTS=seamus.prompt.build_prompts
TASKS=(report-only combined)
FORMATS=(event-only text-only text-with-event text-with-schema)
OUTPUT_DIR=/brtx/603-nvme1/wgantt/SEAMuS/prompt/saved_prompts/
REPORT_ONLY_CORRUPTED_INPUT_DIR=/brtx/603-nvme1/wgantt/SEAMuS/analysis/annotation_corruption/report_edits/test/
REPORT_ONLY_CORRUPTED_EXAMPLES_DIR=/brtx/603-nvme1/wgantt/SEAMuS/analysis/annotation_corruption/report_edits/train/
REPORT_ONLY_CORRUPTED_DATA_FILES=("report_edits_0.10_corrupted.json" "report_edits_0.20_corrupted.json" "report_edits_0.30_corrupted.json" "report_edits_0.40_corrupted.json" "report_edits_0.50_corrupted.json")
REPORT_ONLY_CORRUPTED_OUTPUT_FILES=("report-only_text-with-event_test-few-shot-corrupted-0.10.jsonl" "report-only_text-with-event_test-few-shot-corrupted-0.20.jsonl" "report-only_text-with-event_test-few-shot-corrupted-0.30.jsonl" "report-only_text-with-event_test-few-shot-corrupted-0.40.jsonl" "report-only_text-with-event_test-few-shot-corrupted-0.50.jsonl")
COMBINED_CORRUPTED_INPUT_DIR=/brtx/603-nvme1/wgantt/SEAMuS/analysis/annotation_corruption/combined_edits/test/
COMBINED_CORRUPTED_EXAMPLES_DIR=/brtx/603-nvme1/wgantt/SEAMuS/analysis/annotation_corruption/combined_edits/train/
COMBINED_CORRUPTED_DATA_FILES=("combined_edits_0.10_corrupted.json" "combined_edits_0.20_corrupted.json" "combined_edits_0.30_corrupted.json" "combined_edits_0.40_corrupted.json" "combined_edits_0.50_corrupted.json")
COMBINED_CORRUPTED_OUTPUT_FILES=("combined_text-with-event_test-few-shot-corrupted-0.10.jsonl" "combined_text-with-event_test-few-shot-corrupted-0.20.jsonl" "combined_text-with-event_test-few-shot-corrupted-0.30.jsonl" "combined_text-with-event_test-few-shot-corrupted-0.40.jsonl" "combined_text-with-event_test-few-shot-corrupted-0.50.jsonl")

for TASK in ${TASKS[@]}; do
	for FORMAT in ${FORMATS[@]}; do
		if [ "$FORMAT" == "event-only" ]; then
			python -m $BUILD_PROMPTS -t $TASK -f $FORMAT
		else
			python -m $BUILD_PROMPTS -t $TASK -f $FORMAT --use-source-overrides
		fi
	done
done

# Also make few-shot prompts (currently only supported for text-with-event format)
python -m $BUILD_PROMPTS -t report-only -f text-with-event --use-source-overrides --do-few-shot
python -m $BUILD_PROMPTS -t combined -f text-with-event --use-source-overrides --do-few-shot

# Make few-shot prompts with corrupted data
for i in {0..4}; do
	python -m $BUILD_PROMPTS \
		-t report-only \
		-f text-with-event \
		--use-source-overrides \
		--do-few-shot \
		--corrupted \
		--input-file $REPORT_ONLY_CORRUPTED_INPUT_DIR/${REPORT_ONLY_CORRUPTED_DATA_FILES[$i]} \
		--output-file $OUTPUT_DIR/${REPORT_ONLY_CORRUPTED_OUTPUT_FILES[$i]} \
		--few-shot-data-file $REPORT_ONLY_CORRUPTED_EXAMPLES_DIR/${REPORT_ONLY_CORRUPTED_DATA_FILES[$i]}
	
	python -m $BUILD_PROMPTS \
		-t combined \
		-f text-with-event \
		--use-source-overrides \
		--do-few-shot \
		--corrupted \
		--input-file $COMBINED_CORRUPTED_INPUT_DIR/${COMBINED_CORRUPTED_DATA_FILES[$i]} \
		--output-file $OUTPUT_DIR/${COMBINED_CORRUPTED_OUTPUT_FILES[$i]} \
		--few-shot-data-file $COMBINED_CORRUPTED_EXAMPLES_DIR/${COMBINED_CORRUPTED_DATA_FILES[$i]}
done