# Train AlphaQCM csi300
python train_qcm.py \
    --instruments csi300 \
    --pool 20 \
    --seed 0

# Evaluate AlphaQCM results csi300
python run_adaptive_combination.py \
    --expressions_file output/qcm_csi300 \
    --instruments csi300 \
    --cuda 0 \
    --train_end_year 2021 \
    --seed 0 \
    --use_weights True


# Train AlphaQCM csi500
python train_qcm.py \
    --instruments csi500 \
    --pool 20 \
    --seed 0

# Evaluate AlphaQCM results csi500
python run_adaptive_combination.py \
    --expressions_file output/qcm_csi500 \
    --instruments csi500 \
    --cuda 0 \
    --train_end_year 2021 \
    --seed 0 \
    --use_weights True


# Train AlphaGen with PPO
python train_ppo.py \
    --instruments csi300 \
    --pool 20 \
    --seed 0

# Evaluate AlphaGen results  
python run_adaptive_combination.py \
    --expressions_file output/ppo_csi300 \
    --instruments csi300 \
    --cuda 0 \
    --train_end_year 2021 \
    --seed 0 \
    --use_weights True

# Train AlphaGen with PPO
python train_ppo.py \
    --instruments csi500 \
    --pool 20 \
    --seed 0

# Evaluate AlphaGen results  
python run_adaptive_combination.py \
    --expressions_file output/ppo_csi500 \
    --instruments csi300 \
    --cuda 0 \
    --train_end_year 2021 \
    --seed 0 \
    --use_weights True

# Generate Alpha Pool with GFlowNets  CSI300
CUDA_VISIBLE_DEVICES=0 python train_gfn.py \
        --seed 0 \
        --instrument csi300 \
        --pool_capacity 50 \
        --log_freq 500 \
        --update_freq 64 \
        --n_episodes 10000 \
        --encoder_type gnn \
        --entropy_coef 0.01 \
        --entropy_temperature 1.0 \
        --mask_dropout_prob 1.0 \
        --ssl_weight 1.0 \
        --nov_weight 0.3 \
        --weight_decay_type linear \
        --final_weight_ratio 0.0

#  Evaluate and Combine Alpha Pool  CSI300
# out/test_sp500_2020_0/csv_zoo_final.csv
CUDA_VISIBLE_DEVICES=0 python run_adaptive_combination.py \
    --expressions_file /root/AlphaSAGE/data/gfn_logs/pool_50/gfn_gnn_csi300_50_0-0.01-1.0-1.0-1.0-0.3-linear-0.0/pool_9999.json \
    --instruments csi300 \
    --threshold_ric 0.015 \
    --threshold_ricir 0.15 \
    --chunk_size 400 \
    --window inf \
    --n_factors 50 \
    --cuda 0 \
    --train_end_year 2021 \
    --seed 0


# Generate Alpha Pool with GFlowNets  CSI500
CUDA_VISIBLE_DEVICES=0 python train_gfn.py \
        --seed 0 \
        --instrument csi500 \
        --pool_capacity 50 \
        --log_freq 500 \
        --update_freq 64 \
        --n_episodes 10000 \
        --encoder_type gnn \
        --entropy_coef 0.01 \
        --entropy_temperature 1.0 \
        --mask_dropout_prob 1.0 \
        --ssl_weight 1.0 \
        --nov_weight 0.3 \
        --weight_decay_type linear \
        --final_weight_ratio 0.0

#  Evaluate and Combine Alpha Pool  CSI500
# out/test_sp500_2020_0/csv_zoo_final.csv
CUDA_VISIBLE_DEVICES=0 python run_adaptive_combination.py \
    --expressions_file output/gfn_csi500 \
    --instruments csi500 \
    --threshold_ric 0.015 \
    --threshold_ricir 0.15 \
    --chunk_size 400 \
    --window inf \
    --n_factors 50 \
    --cuda 2 \
    --train_end_year 2021 \
    --seed 0 \