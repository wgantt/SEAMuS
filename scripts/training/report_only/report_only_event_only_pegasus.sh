#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1
#SBATCH --job-name=report-only-event-only-pegasus
#SBATCH --time=24:0:0

MODEL="google/pegasus-large"
TASK="report-only"
FORMAT="event_only"
SEED=1337
OUTPUT_DIR="/brtx/603-nvme1/wgantt/SEAMuS-experiments/$TASK-$FORMAT-pegasus-$SEED"

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
	--seed $SEED
