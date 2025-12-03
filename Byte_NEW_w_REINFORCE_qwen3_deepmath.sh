export RAY_DEDUP_LOGS=0
set -xeuo pipefail
pip list
export WANDB_MODE=online
export WANDB_BASE_URL="https://api.bandw.top"
export WANDB_API_KEY=652495ce2365c4a81500574ad4bcee83f91ee48d
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
project_name='Internal_Policy_Deepmath1029'


MODEL_PATH=${MODEL_PATH:-"/mnt/hdfs/tanyuqiao/models/models/Qwen3-4B"}
# CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/verl/checkpoints/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"/opt/tiger/RLVR-Decomposed/data/deepmath-5k.parquet"}
TEST_FILE=${TEST_FILE:-["/opt/tiger/RLVR-Decomposed/data/aime_2024.parquet","/opt/tiger/RLVR-Decomposed/data/aime_2025.parquet","/opt/tiger/RLVR-Decomposed/data/amc2023.parquet","/opt/tiger/RLVR-Decomposed/data/math500.parquet"]}
use_dynamic_bsz=True
actor_ppo_max_token_len=$((1024 * 32))
infer_ppo_max_token_len=$((1024 * 32))
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout


# advantage="positive"   # PSR
# advantage="negative"   # NSR
advantage="weighted"   # W-REINFORCE
positive_advantage_weight=0.1   # For W-REINFORCE only
prompt_template_type="qwen3_no_thinking"
max_prompt_length=$((1024 * 1))
loss_agg_mode="token-mean"
max_response_length=$((1024 * 8))
kl_coef=0.0
lr=1e-6
experiment_name="Qwen3-4b_byte_w_reinforce_deepmath5k_bsz128_mini32_n8_resp8k_higherclip0.2_lr1e-6$"


ray job submit --runtime-env=verl/trainer/runtime_env.yaml --no-wait -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=psr_nsr \
    algorithm.advantage=$advantage \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.prompt_template_type=$prompt_template_type \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.default_local_dir=/mnt/hdfs/tanyuqiao/entropy_grad/checkpoints/${project_name}/${experiment_name} \
    trainer.experiment_name=${experiment_name} \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb',"console"] \
    trainer.project_name="${project_name}" \
    trainer.n_gpus_per_node=8 \
    +trainer.val_before_train=True \
    trainer.val_before_train=True \
    trainer.test_freq=30 \
    trainer.save_freq=50 \
    trainer.total_training_steps=300 $@
    # algorithm.positive_advantage_weight=$positive_advantage_weight \
