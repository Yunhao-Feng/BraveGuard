
conda activate /home/admin/.conda/envs/braveguard
export PATH=/home/admin/.conda/envs/braveguard/bin:$PATH

python rock_runner.py
python run_sft.py \
    --input exports \
    --dataset data/subset.json \
    --mode 3 \
    --model-path model_cache/qwen3_guard_8b \
    --model-type qwen3 \
    --output-dir sft_runs/qwen3_guard_8b \
    --template qwen3 \
    --epochs 50 \
    --learning-rate 2e-5 \
    --warmup-ratio 0.1 \
    --lr-scheduler-type cosine \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 2 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --lora-dropout 0.05 \
    --balance-labels oversample \
    --eval-strategy epoch \
    --save-strategy epoch \
    --save-total-limit 2 \
    --logging-steps 1 \
    --cutoff-len 16000 \
    --export-after-train