#!/bin/bash

# for validation data, we should always get gradients with sgd
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
CKPT=1500 
task="lambada" # tydiqa, mmlu
data_dir="" # path to data
model="" # path to model
output_path="" # path to output
dims=8192 # dimension of projection, can be a list

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi

python3 -m calculation.get_info \
--task $task \
--info_type grads \
--model_path $model \
--output_path $output_path \
--gradient_projection_dimension $dims \
--gradient_type sgd \
--data_dir $data_dir 

wait