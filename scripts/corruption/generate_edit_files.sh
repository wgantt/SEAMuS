#!/bin/bash
SPLITS=(train test)
PROJECT_ROOT=/brtx/603-nvme1/wgantt/SEAMuS/ # CHANGE ME!
OUTPUT_DIR=$PROJECT_ROOT/resources/annotation_corruption/
GENERATE_EDITS=$PROJECT_ROOT/seamus/corruption/generate_edits.py
REPORT_RANDOM_SEED=14620 # DO NOT CHANGE
SOURCE_RANDOM_SEED=14627 # DO NOT CHANGE
Ps=(0.10 0.20 0.30 0.40 0.50)

for SPLIT in ${SPLITS[@]}; do
	echo "----------------------------------"
	echo "Generating edits for split '$SPLIT'"
	echo "----------------------------------"
	for p in ${Ps[@]}; do
		PYTHONPATH=$PROJECT_ROOT python $GENERATE_EDITS $OUTPUT_DIR/report_edits/$SPLIT/report_edits_$p.json --doc report --p $p --split $SPLIT --seed $REPORT_RANDOM_SEED
	done

	for p in ${Ps[@]}; do
		PYTHONPATH=$PROJECT_ROOT python $GENERATE_EDITS $OUTPUT_DIR/source_edits/$SPLIT/source_edits_$p.json --doc source --p $p --split $SPLIT --seed $SOURCE_RANDOM_SEED
	done
done