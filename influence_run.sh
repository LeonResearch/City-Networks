run_models(){
    local DATASET=$1
    local model=$2
    local device_idx=$3 # GPU idx
    local start_core=$4 # Starting CPU index
    local end_core=$5 # end CPU index

    # Experiment configs
    local layers=16 # must match with the name of saved results/models
    local n_hidden=64 # must match with the name of saved results/models
    local epochs=30000 # must match with the name of saved results/models
    local num_samples=10000 # Number of sampled nodes used for calculating influence
    local exp_name='May08' # The experiment name of the saved model.
    local influence_dir='influence_results/testing'

    if [ $DATASET == "cora" ]; then
        local epochs=1000
    elif [ $DATASET == "citeseer" ]; then
        local epochs=1000
    else
        local epochs=30000
    fi

    # Run the Python script with taskset to set CPU affinity
    echo "Start_CPU: $start_core, End_CPU: $end_core Data: $DATASET, Model: $model, Num_layers: $layers"
    taskset -c $start_core-$end_core python influence_main.py --method $model --device $device_idx \
    --dataset $DATASET --num_layers $layers --hidden_channels $n_hidden --num_samples_influence $num_samples\
    --epochs $epochs --experiment_name $exp_name --influence_dir $influence_dir
    wait
}

# gcn sage cheb sgformer
method=sgformer

# Uncomment to execute in parallel
DATASETS=(
    "cora"
    #"citeseer"
    #"paris"
    #"shanghai"
    #"la"
    #"london"
)
start_core=0 # Starting CPU core
k=6 # Number of CPU cores per job
gpu=0 # CUDA device id
for (( idx=0; idx<${#DATASETS[@]}; idx++ )); do
    end_core=$((start_core + k - 1))
    run_models ${DATASETS[idx]} $method $gpu $start_core $end_core &
    start_core=$((end_core + 1))
done