#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --gpus=1
#SBATCH --job-name=combined-inference
#SBATCH --time=24:0:0

PROJECT_ROOT=/brtx/603-nvme1/wgantt/SEAMuS
DEFAULT_SOURCE_OVERRIDE=$PROJECT_ROOT/resources/saved_contexts/bm25_test_concat_7.json

MODEL_PATH=$1 # Path to trained model
HUB_NAME=$2 # Model name as it appears on the HuggingFace hug (e.g. facebook/bart-large)
DATA_PATH=$3 # SEAMuS/data/{train.json, dev.json, test.json}
FORMAT=$4 # event_only, text_only, text_with_schema, text_with_event

export CONDAROOT=/home/wgantt/miniconda3
export PATH=$CONDAROOT/condabin:$PATH
export PYTHONPATH="$PYTHONPATH:."
source $HOME/.bashrc
export MKL_THREADING_LAYER=GNU
conda activate seamus 
python -m seamus.inference.inference $1 $2 \
	--data-path $3 \
	--task combined \
	--batch-size 4 \
	--input-format $4 \
	--source-override-path $DEFAULT_SOURCE_OVERRIDE