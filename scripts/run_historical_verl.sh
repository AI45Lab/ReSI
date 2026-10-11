#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
VERL=${R2AI_VERL_ROOT:-"$ROOT/.runtime/verl"}
BASE=${BASE:?Set BASE to the standalone parent model}
MODEL_FAMILY=${MODEL_FAMILY:-qwen36}
ADAPTER=${ADAPTER:-null}
RUN=${RUN:-"$ROOT/runs/grpo/online_verl_2step_seed42"}
DATA=${DATA:-"$RUN/prompts.parquet"}
VAL_DATA=${VAL_DATA:-"$DATA"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
ACTOR_LR=${ACTOR_LR:-1e-6}
TOTAL_STEPS=${TOTAL_STEPS:-2}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
SAVE_FREQ=${SAVE_FREQ:--1}
SAVE_STEPS=${SAVE_STEPS:-}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
RESUME_MODE=${RESUME_MODE:-disable}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen36_mixed_c2_real_2step_seed42}
WANDB_MODE=${WANDB_MODE:-offline}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
TEST_FREQ=${TEST_FREQ:--1}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-8}
MAX_NUM_GEN_BATCHES=${MAX_NUM_GEN_BATCHES:-8}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
ROLLOUT_ENABLE_PREFIX_CACHING=${ROLLOUT_ENABLE_PREFIX_CACHING:-False}
TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE:-8}
TRAINER_NNODES=${TRAINER_NNODES:-1}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-False}

grep -Fq 'max_num_gen_batches=max_num_gen_batches' \
  "$VERL/verl/trainer/ppo/v1/trainer_base.py" || {
  echo "VERL V1 finite DAPO cap is missing; apply patches/verl_historical_runtime.patch" >&2
  exit 2
}
grep -Fq 'Sync DAPO refill budget exhausted' \
  "$VERL/verl/trainer/ppo/v1/replay_buffer.py" || {
  echo "VERL V1 finite DAPO cap is incomplete; apply patches/verl_historical_runtime.patch" >&2
  exit 2
}
grep -Fq 'final_output.extra_fields.update(output.extra_fields)' \
  "$VERL/verl/workers/rollout/llm_server.py" || {
  echo "VERL completion metadata patch is missing; rerun scripts/setup_verl.sh" >&2
  exit 2
}
grep -Fq 'self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)' \
  "$VERL/verl/experimental/reward_loop/reward_manager/naive.py" || {
  echo "VERL final-answer reward decoding patch is missing; rerun scripts/setup_verl.sh" >&2
  exit 2
}
test -s "$DATA"
if [[ "$ADAPTER" != null ]]; then test -s "$ADAPTER/adapter_model.safetensors"; fi
mkdir -p "$RUN/checkpoints"

export R2AI_SAVE_STEPS="$SAVE_STEPS"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

cd "$VERL"
family_overrides=()
case "$MODEL_FAMILY" in
  qwen36)
    family_overrides+=(
      +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
      +actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode=none
    )
    ;;
  ministral3)
    family_overrides+=(
      +actor_rollout_ref.rollout.engine_kwargs.vllm.tokenizer_mode=mistral
      +actor_rollout_ref.rollout.engine_kwargs.vllm.config_format=mistral
      +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=mistral
      +actor_rollout_ref.rollout.engine_kwargs.vllm.trust_remote_code=True
    )
    ;;
  *) echo "unsupported MODEL_FAMILY=$MODEL_FAMILY" >&2; exit 2 ;;
esac
exec .venv/bin/python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  algorithm.filter_groups.enable=True \
  algorithm.filter_groups.metric=acc \
  algorithm.filter_groups.max_num_gen_batches="$MAX_NUM_GEN_BATCHES" \
  algorithm.filter_groups.max_inflight_gen_batches=1 \
  data.train_files="$DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.seed="${SEED:-42}" \
  +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING:-true}" \
  actor_rollout_ref.model.path="$BASE" \
  actor_rollout_ref.model.lora_adapter_path="$ADAPTER" \
  actor_rollout_ref.model.lora_rank="$LORA_RANK" \
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
  "actor_rollout_ref.model.target_modules=${LORA_TARGETS:-all-linear}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LEN" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_LOW:-0.2}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_HIGH:-0.28}" \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  +actor_rollout_ref.actor.checkpoint.save_lora_only=True \
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.enable_prefix_caching="$ROLLOUT_ENABLE_PREFIX_CACHING" \
  actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
  "${family_overrides[@]}" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.top_p="${TOP_P:-0.95}" \
  actor_rollout_ref.rollout.top_k="${TOP_K:--1}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_MODEL_LEN" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAX_MODEL_LEN" \
  actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
  reward.reward_manager.name=naive \
  reward.custom_reward_function.path="$ROOT/src/r2ai_autoresearch/training/verl_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.logger="$TRAINER_LOGGER" \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=sync \
  trainer.v1.sampler.sync_refill_failed_groups=True \
  trainer.project_name=r2ai_online_grpo \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$TRAINER_N_GPUS_PER_NODE" \
  trainer.nnodes="$TRAINER_NNODES" \
  trainer.default_local_dir="$RUN/checkpoints" \
  trainer.rollout_data_dir="$RUN/rollouts" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  +trainer.save_final_checkpoint=True \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.resume_mode="$RESUME_MODE" \
  "$@"
