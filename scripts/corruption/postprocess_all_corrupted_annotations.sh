#!/bin/bash
PROJECT_ROOT=/brtx/603-nvme1/wgantt/SEAMuS # CHANGE ME!
SPLITS=(train test)
ANNOTATION_CORRUPTION_DIR=$PROJECT_ROOT/resources/annotation_corruption
REPORT_EDITS_DIR=$ANNOTATION_CORRUPTION_DIR/report_edits
SOURCE_EDITS_DIR=$ANNOTATION_CORRUPTION_DIR/source_edits

POSTPROCESS=$PROJECT_ROOT/seamus/corruption/postprocess_corrupted_annotations.py

CORRUPTED_REPORT_RESPONSE_FILES=("report_edits_0.10_response.jsonl" "report_edits_0.20_response.jsonl" "report_edits_0.30_response.jsonl" "report_edits_0.40_response.jsonl" "report_edits_0.50_response.jsonl")
CORRUPTED_REPORT_OUTPUT_FILES=("report_edits_0.10_corrupted.json" "report_edits_0.20_corrupted.json" "report_edits_0.30_corrupted.json" "report_edits_0.40_corrupted.jsonl" "report_edits_0.50_corrupted.jsonl")

CORRUPTED_SOURCE_RESPONSE_FILES=("source_edits_0.10_response.jsonl" "source_edits_0.20_response.jsonl" "source_edits_0.30_response.jsonl" "source_edits_0.40_response.jsonl" "source_edits_0.50_response.jsonl")
CORRUPTED_SOURCE_OUTPUT_FILES=("source_edits_0.10_corrupted.json" "source_edits_0.20_corrupted.json" "source_edits_0.30_corrupted.json" "source_edits_0.40_corrupted.jsonl" "source_edits_0.50_corrupted.jsonl")

for SPLIT in ${SPLITS[@]}; do
	echo "----------------------------------------------"
	echo "Postprocessing REPORT files for split '$SPLIT'"
	echo "----------------------------------------------"
	for i in ${!CORRUPTED_REPORT_RESPONSE_FILES[@]}; do
		PYTHONPATH=$PROJECT_ROOT python $POSTPROCESS $REPORT_EDITS_DIR/$SPLIT/${CORRUPTED_REPORT_RESPONSE_FILES[$i]} $REPORT_EDITS_DIR/$SPLIT/${CORRUPTED_REPORT_OUTPUT_FILES[$i]} -d report --split $SPLIT
	done

	echo "----------------------------------------------"
	echo "Postprocessing SOURCE files for split '$SPLIT'"
	echo "----------------------------------------------"
	for i in ${!CORRUPTED_SOURCE_RESPONSE_FILES[@]}; do
		PYTHONPATH=$PROJECT_ROOT python $POSTPROCESS $SOURCE_EDITS_DIR/$SPLIT/${CORRUPTED_SOURCE_RESPONSE_FILES[$i]} $SOURCE_EDITS_DIR/$SPLIT/${CORRUPTED_REPORT_OUTPUT_FILES[$i]} -d source --split $SPLIT
	done

done
