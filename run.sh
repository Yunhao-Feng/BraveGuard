conda activate /home/admin/.conda/envs/braveguard
export PATH=/home/admin/.conda/envs/braveguard/bin:$PATH

# python run_sft.py \
#     --input exports \
#     --dataset data/subset.json \
#     --mode 3 \
#     --model-path model_cache/qwen3_guard_8b \
#     --model-type qwen3 \
#     --output-dir sft_runs/qwen3_guard_8b \
#     --template qwen3 \
#     --dry-run


python run_sft.py \
    --input exports_v8 \
    --dataset data/subset.json \
    --mode 3 \
    --model-path model_cache/qwen3_guard_8b \
    --model-type qwen3 \
    --output-dir sft_runs/qwen3_guard_8b \
    --template qwen3 \
    --epochs 30 \
    --learning-rate 1e-5 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --cutoff-len 16000 \
    --export-after-train


python run_eval.py \
    --input exports_v8 \
    --model-paths sft_runs/qwen3_guard_8b/merged \
    --model-type qwen3 \
    --prompt-style sft_flat \
    --mode 3 \
    --output-dir guard_sft

python run_eval.py \
    --input exports_v8 \
    --model-paths model_cache/qwen3_guard_8b \
    --model-type qwen3 \
    --prompt-style sft_flat \
    --mode 3 \
    --output-dir guard