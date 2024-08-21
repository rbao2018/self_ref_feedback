current_time=$(date +%s)
one_hour_ago=$((current_time - 3600))
for gpu in $(seq 0 7); do
  last_allocated_time[$gpu]=$one_hour_ago
done

model_name_or_path="/mnt/data/huggingface/models/Llama-2-7b-chat-hf"
model_class="llama2"
while true;
do
  gpu_allocated=false
  for gpu in $(seq 0 7); do
      gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu)
      # 获取当前时间戳
      current_time=$(date +%s)
      # 检查上次分配任务的时间戳，如果在五分钟内，则跳过此GPU
      last_allocated_diff=$((current_time - ${last_allocated_time[$gpu]}))
      if [[ $last_allocated_diff -lt 180 ]]; then
          continue
      elif [[ $gpu_memory -lt 400 ]]; then
        current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
        echo "[$current_datetime] 部署任务被调度到 GPU $gpu 上"
        export CUDA_VISIBLE_DEVICES=$gpu
        lmdeploy serve api_server $model_name_or_path --backend turbomind --model-format hf --server-name 0.0.0.0 --server-port 1008$gpu --model-name llama2 --cache-max-entry-count 0.95 --tp 1 --session-len 8192 > /tmp/deploy_$gpu.log 2>&1 &
        gpu_allocated=true
        # 更新GPU的最后一次分配任务的时间戳
        last_allocated_time[$gpu]=$current_time
        break
      fi
  done
  if [[ "$gpu_allocated" = "false" ]]; then
    current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$current_datetime] 没有服务终止..."
    sleep 60
  fi
  sleep 8
done