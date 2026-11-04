UV_VENV_PATH=/path/to/venv # Change it with your env

MODEL_PATH=/path/to/Qwen/Qwen3-Omni-30B-A3B-Instruct
INPUT_PATH=/path/to/prepared/data

# Check whether the vlmm_server is active


source $UV_VENV_PATH/bin/activate


#task="MT" ; dataset="ACL6060" ; evalset="speechM-v1"
task="MT" ; dataset="FLORES" ; evalset="speechM-v1"
#task="ASR" ; dataset="ACL6060" ; evalset="speechM-v1"
#task="ASR" ; dataset="COVOST" ; evalset="speechM-v1"
#task="ASR" ; dataset="LRS3" ; evalset="speechM-v1"
#task="ASR" ; dataset="MLS" ; evalset="speechM-v1"
#task="ASR" ; dataset="MTEDX" ; evalset="speechM-v1"
#task="ASR" ; dataset="MUSTC" ; evalset="speechM-v1"

NUM_CONCURRENT=4
OUTPUT_PATH="output/speechM-v1/metadata/${task}/${dataset}"
INPUT_DATA_DIR=${INPUT_PATH}
INPUT_JSON_FILE=${INPUT_PATH}/${evalset}/metadata/${task}/${dataset}/${dataset}.jsonl 

OUTPUT_JSON_FILE="output/${evalset}/metadata/${task}/${dataset}"
OUTPUT_DATA_DIR="output"
OUTPUT_RELATIVE_DIR="${evalset}/audio/${task}/${dataset}"

# other options:
#   --port PORT [default: localhost]
#   --host HOST [host: 8091]
#   --modalities MODALITIES [default: None]

python run_vllm_client_multiple_queries.py \
--num-concurrent-requests ${NUM_CONCURRENT} \
--model ${MODEL_PATH} \
--input-json-file ${INPUT_JSON_FILE} \
--input-data-dir ${INPUT_DATA_DIR} \
--output-json-file ${OUTPUT_JSON_FILE} \
--output-data-dir ${OUTPUT_DATA_DIR} \
--output-relative-dir ${OUTPUT_RELATIVE_DIR} \
&> LOG.run_vllm_client_multiple_queries_${task}_${dataset}


