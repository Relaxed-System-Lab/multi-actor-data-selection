#!/bin/bash


# forlambada 
CKPT=1500
output_path="/"
gradient_path=""

validation_gradient_path=""
target_task_names="lambada"

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

python3 -m calculation.matching \
--gradient_path $gradient_path \
--train_file_names $train_file_names \
--ckpts $ckpts \
--checkpoint_weights $checkpoint_weights \
--validation_gradient_path $validation_gradient_path \
--target_task_names $target_task_names \
--output_path $output_path