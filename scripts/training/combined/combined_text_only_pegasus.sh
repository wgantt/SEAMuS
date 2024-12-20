#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1
#SBATCH --job-name=combined-text-only-pegasus
#SBATCH --time=24:0:0

MODEL="google/pegasus-large"
TASK="combined"
FORMAT="text_only"
SEED=1337
RETRIEVAL_SETTING="bm25_concat_7"
OUTPUT_DIR="/brtx/603-nvme1/wgantt/SEAMuS-experiments/$TASK-$FORMAT-$RETRIEVAL_SETTING-pegasus-$SEED"

# These retrieved contexts are used in place of
# the full, original source documents
SRC_OVERRIDE_DIR="/brtx/603-nvme1/wgantt/SEAMuS/resources/saved_contexts/"
TRAIN_SRC_OVERRIDE=$SRC_OVERRIDE_DIR/bm25_train_concat_7.json
DEV_SRC_OVERRIDE=$SRC_OVERRIDE_DIR/bm25_dev_concat_7.json
TEST_SRC_OVERRIDE=$SRC_OVERRIDE_DIR/bm25_test_concat_7.json

export CONDAROOT=/home/wgantt/miniconda3
export PATH=$CONDAROOT/condabin:$PATH
export PYTHONPATH="$PYTHONPATH:."
source $HOME/.bashrc
export MKL_THREADING_LAYER=GNU
conda activate seamus 
python -m seamus.training.train \
	$OUTPUT_DIR \
	-m $MODEL \
	-t $TASK \
	--input-format $FORMAT \
	--source-override-path-train $TRAIN_SRC_OVERRIDE \
	--source-override-path-dev $DEV_SRC_OVERRIDE \
	--source-override-path-test $TEST_SRC_OVERRIDE \
	--seed $SEED
