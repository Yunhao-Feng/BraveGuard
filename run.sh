python generate.py 

python rock_runner.py
conda activate /home/admin/.conda/envs/braveguard
export PATH=/home/admin/.conda/envs/braveguard/bin:$PATH

# 关键点：
# 1) SFT / eval 都使用 sft_flat + plain chat template，让模型学习“扫描 Agent 轨迹是否有害”，
#    避免 Qwen3Guard tokenizer 自带模板继续把任务改写成“判断最后一个 USER query 是否违禁”。
# 2) 之前 trainer_log 只有 12 个 optimizer update，loss 仍在 2 左右，属于明显欠拟合；
#    因此降低 gradient_accumulation、增加 epoch，并用 epoch 级 eval/save 选择最佳 checkpoint。
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

python run_eval.py \
    --input exports_test \
    --model-paths sft_runs/qwen3_guard_8b/merged \
    --model-type qwen3 \
    --prompt-style sft_flat \
    --chat-template plain \
    --mode 3 \
    --output-dir guard_sft

python run_eval.py \
    --input exports_test \
    --model-paths model_cache/qwen3_guard_8b \
    --model-type qwen3 \
    --prompt-style sft_flat \
    --chat-template plain \
    --mode 3 \
    --output-dir guard


nohup bash rollout.sh >> rollout.txt 2>&1 &
635866
