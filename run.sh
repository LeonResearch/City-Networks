run_models(){
    local DATASET=$1
    local model=$2
    local device_idx=$3 # GPU idx
    local start_core=$4 # Starting CPU/GPU index
    local k=$5 # Number of CPU cores to use for each model

    # Experiment configs
    local layers=(16)
    local hidden_size=64
    local runs=1
    local exp_name='final'

    # Loop through #layer and execute with pre-specified CPU/GPU
    for (( idx=0; idx<${#layers[@]}; idx++ )); do
        local end_core=$((start_core + k - 1))
        local num_layers=${layers[idx]}

        echo "Device: $device_idx | Data: $DATASET | Model: $model | Num_layers: $num_layers"
        # The results will save to ./results/exp_name
        taskset -c $start_core-$end_core python train.py --exp_name $exp_name \
        --method $model --dataset $DATASET --num_layers $num_layers --hidden_size $hidden_size \
        --device $device_idx --runs $runs
    done
}

# Unomment to execute in parallel
DATASETS=(
    #paris
    #shanghai
    la
    #london
)

# Given a machine with 8 GPUs and 80 cores,
# the tasks can be executed in parallel
for data in ${DATASETS[@]}; do
    run_models $data gcn 0 0 10 &
    #run_models $data sage 1 10 10 &
    #run_models $data gat 2 20 10 &
    #run_models $data gcnii 3 30 10 &
    #run_models $data cheb 4 40 10 &
    #run_models $data exphormer 5 50 10 &
    #run_models $data sgformer 6 60 10 &
    #run_models $data mlp 7 70 10 &
done