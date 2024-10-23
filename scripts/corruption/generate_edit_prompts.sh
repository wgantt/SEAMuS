#!/bin/bash

SPLITS=(train test)
PROJECT_ROOT=/brtx/603-nvme1/wgantt/SEAMuS # CHANGE ME!
BUILD_PROMPTS=$PROJECT_ROOT/seamus/corruption/build_template_edit_prompts.py
REPORT_EDITS_DIR=$PROJECT_ROOT/resources/annotation_corruption/report_edits
SOURCE_EDITS_DIR=$PROJECT_ROOT/resources/annotation_corruption/source_edits
Ps=(0.10 0.20 0.30 0.40 0.50)

for SPLIT in ${SPLITS[@]}; do
	echo "------------------------------------------"
	echo "Generating REPORT edits for split '$SPLIT'"
	echo "------------------------------------------"
	for p in ${Ps[@]}; do
		echo "Generating prompts for p=$p"
		PYTHONPATH=. python $BUILD_PROMPTS $SPLIT $REPORT_EDITS_DIR/$SPLIT/report_edits_$p.json $REPORT_EDITS_DIR/$SPLIT/report_edits_${p}_prompts.jsonl --doc report
	done

	echo "------------------------------------------"
	echo "Generating SOURCE edits for split '$SPLIT'"
	echo "------------------------------------------"
	for p in ${Ps[@]}; do
		echo "Generating prompts for p=$p"
		PYTHONPATH=. python $BUILD_PROMPTS $SPLIT $SOURCE_EDITS_DIR/$SPLIT/source_edits_$p.json $SOURCE_EDITS_DIR/$SPLIT/source_edits_${p}_prompts.jsonl --doc source 
	done
done