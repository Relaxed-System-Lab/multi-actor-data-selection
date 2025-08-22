
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

CKPT=3000
train_file="" #
model="" # path to model
output_path="" # path to output
dims=8192 # dimension of projection, can be a list
gradient_type="sgd"

if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi
gpu_index=0
for slc in {0..7}; do
    CUDA_VISIBLE_DEVICES=${gpu_index} \
    python3 -m calculation.get_info \
    --train_file $train_file \
    --info_type grads_train \
    --model_path $model \
    --output_path $output_path \
    --gradient_projection_dimension $dims \
    --gradient_type $gradient_type \
    --max_samples 10000 \
    --slices $slc 

    ((gpu_index=(gpu_index+1)%8))
done

wait