#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1
#SBATCH --job-name=report-only-text-schema-t5
#SBATCH --time=24:0:0

MODEL="t5-large"
TASK="report-only"
FORMAT="text_with_schema"
SEED=1337
OUTPUT_DIR="/brtx/603-nvme1/wgantt/SEAMuS-experiments/$TASK-$FORMAT-t5-$SEED"

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
	--seed $SEED \
	--gradient-checkpointing