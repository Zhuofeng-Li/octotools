#!/bin/bash

############ [1] Batching Run ############
PROJECT_DIR="./"

############
LABEL="gpt4o_baseline"

THREADS=8
TASK="bamboogle"
DATA_FILE="$TASK/data/data.json"
LOG_DIR="$TASK/logs/$LABEL"
OUT_DIR="$TASK/results/$LABEL"
CACHE_DIR="$TASK/cache"

LLM="gpt-4o"

ENABLED_TOOLS="Web_Agent_Tool,Generalist_Solution_Generator_Tool"

cd $PROJECT_DIR

RESPONSE_TYPE="base_response"
python $TASK/calculate_score.py \
--data_file $DATA_FILE \
--result_dir $OUT_DIR \
--response_type $RESPONSE_TYPE \
--output_file "final_results_$RESPONSE_TYPE.json" \
| tee "$OUT_DIR/final_score_$RESPONSE_TYPE.log"

