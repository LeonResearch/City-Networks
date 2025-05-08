run_models(){
    local DATASET=$1
    local model=$2
    local device_idx=$3 # GPU idx
    local start_core=$4 # Starting CPU index
    local end_core=$5 # end CPU index

    # Experiment configs
    local layers=16
    local n_hidden=64
    local epochs=30000
    local num_samples=10000
    local exp_name='May08'
    local influence_dir='influence_results/May03'

    # Run the Python script with taskset to set CPU affinity
    echo "Start_CPU: $start_core, End_CPU: $end_core Data: $DATASET, Model: $model, Num_layers: $layers"
    taskset -c $start_core-$end_core python influence_main.py --method $model --device $device_idx \
    --dataset $DATASET --num_layers $layers --hidden_channels $n_hidden --num_samples_influence $num_samples\
    --epochs $epochs --experiment_name $exp_name --influence_dir $influence_dir
    wait
}

# gcn sage cheb sgformer
method=gcn

DATASETS=(
    "paris"
    #"shanghai"
    #"la"
    "london"
)
start_core=10 # Starting CPU core
k=10 # Number of CPU cores per job
gpu=0 # CUDA device id
for (( idx=0; idx<${#DATASETS[@]}; idx++ )); do
    end_core=$((start_core + k - 1))
    run_models ${DATASETS[idx]} $method $gpu $start_core $end_core &
    start_core=$((end_core + 1))
done