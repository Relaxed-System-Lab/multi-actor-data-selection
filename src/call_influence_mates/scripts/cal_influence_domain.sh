#!/bin/bash

MODEL_NAME="/path/to/model"
RESUME="/MATES/huggingface_checkpoint_2/pytorch_model.bin"


gpu_index=0


for slc in {0..7}; do
    echo "Running slice $slc on GPU $gpu_index of Domain"   
    
    CUDA_VISIBLE_DEVICES=${gpu_index} \
    python ../cal_influence.py \
    --base_directory "/path/to/folder"\
    --model_name "$MODEL_NAME" \
    --out_dir "/path/to/folder/domain/influence_${slc}.jsonl" \
    --slice $slc \
    --resume "$RESUME" \
    > /path/to/folder/logs/log_job_domain_${slc}_gpu${gpu_index}_.out 2>&1 &
    
    ((gpu_index=(gpu_index+1)%8))
done

wait