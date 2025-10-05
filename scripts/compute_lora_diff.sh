EXP=lora_rank_1_iterations_400_lr_3e-05_start_guidance_9.0_negative_guidance_2.0_ddim_steps_50_hyper
HYPER_LORA_DIR=/data/pwojcik/UnGuide/output/$EXP/models/


python compute_lora_diff.py \
  --config configs/stable-diffusion/v1-inference.yaml \
  --ckpt models/sd-v1-4.ckpt \
  --lora "${HYPER_LORA_DIR}hyper_lora.pth" \
  --prompts_json data/cat.json \
  --output_dir output/$EXP/ \
  --device cuda:0 \
  --seed 42 \
  --ddim_steps 50 \
  --t_enc 40 \
  --repeats 30 \
  --n_samples 1