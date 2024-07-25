NNODES=$1
DATASET=$2
PROBS=$3
BS=$4
LR=$5
LOGDIR=$6
PREFIX=$7

if [ "$LOGDIR" == "" ]; then
    LOGDIR=/mnt/data/output
fi

if [ "$PREFIX" == "" ]; then
    PREFIX=test
fi
if [ "$NNODES" == "1" ]; then
    MASTER_ADDR=localhost
    RANK=0
fi
mkdir -p $LOGDIR/$PREFIX

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export MAX_JOBS=32
export MAX_SEQ_LEN=2048
export NCCL_ALGO=Tree

now_date=$(date +%Y_%m%d_%H%M)

torchrun --nproc_per_node 8 --nnodes $NNODES --master_addr $MASTER_ADDR --master_port 6666 --node_rank $RANK /mnt/data/self_ref_feedback/train_rm_llama2.py \
    --logging_path $LOGDIR/$PREFIX \
    --save_path /mnt/data/temp/output/$PREFIX \
    --save_steps -1 \
    --logging_steps 10 \
    --eval_steps 128 \
    --train_batch_size 256 \
    --critic_train_batch_size $BS \
    --pretrain /mnt/data/huggingface/models/Llama-2-7b-hf \
    --packing_samples \
    --loss logexpwithlm \
    --apply_chat_template \
    --tokenizer_chat_template "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user')!= (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message!= false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]\n' }}{% elif message['role'] == 'assistant' %}{{ content.strip() + eos_token }}{% endif %}{% endfor %}" \
    --prompt_key message \
    --chosen_key chose \
    --rejected_key reject \
    --max_epochs 1 \
    --zero_stage 3 \
    --max_len $MAX_SEQ_LEN \
    --learning_rate $LR \
    --dataset $DATASET \
    --dataset_probs $PROBS \
    --use_wandb \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing >>"$LOGDIR/$PREFIX/$now_date"_train.log 2>&1