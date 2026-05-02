run_models(){
    local DATASET=$1
    local model=$2
    local device_idx=$3 # GPU idx
    local start_core=$4 # Starting CPU index
    local end_core=$5 # end CPU index

    # Experiment configs
    local layers=16 # must match with the name of saved results/models
    local hidden_size=128 # must match with the name of saved results/models
    local num_samples=10000 # Number of sampled nodes used for calculating influence
    local exp_name='iclr' # The experiment name of the saved model.
    local influence_dir='influence_results/iclr'

    # Run the Python script with taskset to set CPU affinity
    echo "Start_CPU: $start_core, End_CPU: $end_core Data: $DATASET, Model: $model, Num_layers: $layers"
    taskset -c $start_core-$end_core python influence_main.py --method $model --device $device_idx \
    --dataset $DATASET --num_layers $layers --num_samples_influence $num_samples --hidden_size $hidden_size \
    --influence_dir $influence_dir --exp_name $exp_name
}

# gcn sage cheb gat gcnii sgformer gps exphormer
METHODS=(
    gcn
    #sage
    #gcnii
    #exphormer
    #sgformer
)

# Uncomment to execute in parallel
DATASETS=(
    paris
    #shanghai
    #la
    #london

)  
start_core=0 # Starting CPU core
k=20 # Number of CPU cores per job
gpu=0 # CUDA device id
for (( idx=0; idx<${#DATASETS[@]}; idx++ )); do
    for (( i=0; i<${#METHODS[@]}; i++ )); do
        end_core=$((start_core + k - 1))
        run_models ${DATASETS[idx]} ${METHODS[i]} $((i)) $start_core $end_core &
        start_core=$((end_core + 1))
    done
done