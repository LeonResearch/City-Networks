run_models(){
    local DATASET=$1
    local model=$2
    local device_idx=$3 # GPU idx
    local start_core=$4 # Starting CPU/GPU index
    local k=$5 # Number of CPU cores to use for each model
    # Experiment configs
    local layers=(16 8 4 2)
    local batch_size=200000
    local n_hidden=64
    local lr=1e-3
    local epochs=1000
    local window=1
    local exp_name='May08'
    # Loop through #layer and execute with pre-specified CPU/GPU
    for (( idx=0; idx<${#layers[@]}; idx++ )); do
        local end_core=$((start_core + k - 1))
        local num_layers=${layers[idx]}

        echo "Data: $DATASET, Model: $model, Num_layers: $num_layers"
        # The results will save to ./results/exp_name
        taskset -c $start_core-$end_core python train.py --device $device_idx --method $model --dataset $DATASET \
        --runs 5 --seed 0 --save_model --experiment_name $exp_name \
        --num_layers $num_layers --hidden_channels $n_hidden --lr $lr --batch_size $batch_size --epochs $epochs\
        --weight_decay 1e-5  --dropout 0.2 --display_step $window \
        --num_heads 1 --transformer_weight_decay 1e-2 --transformer_dropout 0.5
    done
}

# Uncomment to execute in parallel
DATASETS=(
    "cora"
    "citeseer"
)

# Given a machine with 4 GPUs and 40 cores,
# the tasks can be executed in parallel
for data in ${DATASETS[@]}; do
    run_models $data sgformer 0 0 30 &
    #run_models $data sage 1 10 10 &
    #run_models $data cheb 2 20 10 &
    #run_models $data sgformer 3 30 10 &
done