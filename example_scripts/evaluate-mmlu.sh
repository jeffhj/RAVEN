#!/bin/bash
#SBATCH --account=XX
#SBATCH --cpus-per-task=10
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --time=2:00:00
#SBATCH --job-name=evaluate-mmlu
#SBATCH --mem=0
#SBATCH --partition XX
#SBATCH --exclusive

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

size=xl
DATA_DIR="${SHARE_OUTPUT}/data"

port=$(shuf -i 15000-16000 -n 1)

n_shot=1
fusion=0

model=raven
name="mmlu-${model}-${size}-${n_shot}-shot-${fusion}-fusion"

# TRAIN_FILE="${DATA_DIR}/data/mmlu_data/full/train.jsonl"
TRAIN_FILE="${DATA_DIR}/data/mmlu_data/full/short.jsonl"
EVAL_FILES="${DATA_DIR}/data/mmlu_data/full/combined_valid.jsonl ${DATA_DIR}/data/mmlu_data/full/combined_test.jsonl"

PRETRAINED_MODEL=${DATA_DIR}/models/${model}/${size}
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=$SLURM_JOB_ID-${name}
PRECISION="bf16"

TEXTS="${DATA_DIR}/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl"
INFOBOXES="${DATA_DIR}/corpora/wiki/enwiki-dec2021/infobox.jsonl"
PASSAGES="${TEXTS} ${INFOBOXES}"

PRETRAINED_INDEX=${DATA_DIR}/indices/wiki-mlm-${size}-atlas-org

CMD="python ${SHARE_OUTPUT}/source/raven/evaluate.py \
    --model ${model} \
    --n_shots ${n_shot} \
    --fusion ${fusion} \
    --gold_score_mode "pdist" \
    --precision ${PRECISION} \
    --text_maxlength 1024 \
    --target_maxlength 512 \
    --reader_model_type google/t5-${size}-lm-adapt \
    --model_path ${PRETRAINED_MODEL} \
    --train_data ${TRAIN_FILE} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size 2 \
    --n_context 40 --retriever_n_context 40 \
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --write_results \
    --task multiple_choice \
    --multiple_choice_train_permutations all \
    --multiple_choice_eval_permutations cyclic \
    --index_mode flat \
    --load_index_path ${PRETRAINED_INDEX} \
    --passages ${PASSAGES}"

srun -l \
    --no-container-mount-home \
    --output ${SHARE_OUTPUT}/logs/${EXPERIMENT_NAME}_$DATETIME.log ${CMD}