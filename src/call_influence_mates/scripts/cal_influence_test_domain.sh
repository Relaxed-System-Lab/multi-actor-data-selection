#!/bin/bash

MODEL_NAME="/path/to/model"
RESUME="/MATES/huggingface_checkpoint_2/pytorch_model.bin"



gpu_index=0


for slc in {0..7}; do
    echo "Running slice $slc on GPU $gpu_index of Domain"   
    
    # 设置 CUDA_VISIBLE_DEVICES 并运行新的 Python 脚本
    CUDA_VISIBLE_DEVICES=${gpu_index} \
    python ../cal_influence.py \
    --base_directory "/fs-computility/llm/shared/baitianyi/dataset/slimpajama_holdout_shuffle/domain"\
    --model_name "$MODEL_NAME" \
    --out_dir "/fs-computility/llm/shared/baitianyi/dataset/processed2/domain_test/influence_${slc}.jsonl" \
    --slice $slc \
    --resume "$RESUME" \
    >/path/to/folder/logs/log_job_test_domain_${slc}_gpu${gpu_index}_.out 2>&1 &
    
    ((gpu_index=(gpu_index+1)%8))
done

wait