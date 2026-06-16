# Training
/home/karthik/miniconda3/envs/analog-rep/bin/python -u gnn_training/scripts/train.py \
  --config gnn_training/config/full_training_diff_amps_comparators_LDO_opamp_all.yaml \
  --data.data_dir /home/karthik/sim_clean/AMS-opt/netlists

CUDA_VISIBLE_DEVICES=0 /home/karthik/.conda/envs/analog-rep/bin/python -u gnn_training/scripts/train.py \
  --config gnn_training/config/full_training_diff_amps_comparators_LDO_opamp_all.yaml \
  --data.data_dir /home/karthik/sim_clean/AMS-opt/AMS-opt/netlists \
  --training.device cuda

# Visualization
/home/karthik/miniconda3/envs/analog-rep/bin/python -u gnn_training/scripts/analyze_sg_skg_embeddings.py \
  --checkpoint checkpoints_sg_vs_skg/best_model.pt \
  --netlists_root /home/karthik/sim_clean/AMS-opt/netlists \
  --out_dir /home/karthik/sim_clean/AMS-opt/umap_results_full/sg_skg_analysis \
  --export_llmbo_json

In euclid:
/home/karthik/miniconda3/envs/analog-rep/bin/python -u gnn_training/scripts/analyze_sg_skg_embeddings.py \
  --checkpoint checkpoints_sg_vs_skg/best_model.pt \
  --netlists_root /home/karthik/sim_clean/AMS-opt/AMS-opt/netlists \
  --out_dir /home/karthik/sim_clean/AMS-opt/AMS-opt/umap_results_full/sg_skg_analysis \
  --export_llmbo_json

python3 scripts/visualize_embeddings_umap.py   --config gnn_training/config/full_training_diff_amps_comparators_LDO_opamp_all.yaml   --device cpu

# Optimization
cd /home/karthik/sim_clean/AMS-opt && /home/karthik/miniconda3/envs/analog-rep/bin/python LLMBO/llmbo.py   --history 1 --related_mode bottomk --related_k 3 --target_id FC   --gnn_embedding_mode sg

# QA Dataset Generation
cd /home/karthik/sim_clean/AMS-opt
/home/karthik/miniconda3/envs/analog-rep/bin/python filter_with_llmbo.py \
  --circuit FC --paragraph_k -1 --repeats 3 --n_itr 5 --n_proposal_llm 1 --n_proposal_bo 0 \
  --min_delta 0.0 --llm_temperature 0.0

# QA Datset Accuracy Evaluation
CUDA_VISIBLE_DEVICES=3 /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py   --eval_vllm_local --log_timings --enforce_eager   --circuits amp2 FC comp ldo --k 3 --gnn_embedding_mode sg   --cuda_visible_devices 3 --vllm_model /data/karthik/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct   --apply_chat_template --temperature 0 --max_tokens 8 --batch_size 4   --out_jsonl filtered_qa_eval_llama3_8b_local.jsonl   --summary_json filtered_qa_eval_llama3_8b_local_summary.json

# Retrieval ablations: pass output filenames only; eval_filtered_qa.py writes them under QA_Task/out/
CUDA_VISIBLE_DEVICES=6 /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py \
  --eval_vllm_local --log_timings --enforce_eager \
  --circuits amp2 FC comp --k 2 --gnn_embedding_mode sg \
  --related_mode bottomk \
  --cuda_visible_devices 6 --vllm_model llama3_8b \
  --apply_chat_template --temperature 0 --max_tokens 8 --batch_size 4 \
  --out_jsonl filtered_qa_eval_llama3_8b_bottomk.jsonl \
  --summary_json filtered_qa_eval_llama3_8b_bottomk_summary.json

CUDA_VISIBLE_DEVICES=6 /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py \
  --eval_vllm_local --log_timings --enforce_eager \
  --circuits amp2 FC comp --k 2 --gnn_embedding_mode sg \
  --related_mode random --random_seed 0 \
  --cuda_visible_devices 6 --vllm_model llama3_8b \
  --apply_chat_template --temperature 0 --max_tokens 8 --batch_size 4 \
  --out_jsonl filtered_qa_eval_llama3_8b_random.jsonl \
  --summary_json filtered_qa_eval_llama3_8b_random_summary.json

CUDA_VISIBLE_DEVICES=6 /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py   --eval_vllm_local --log_timings --enforce_eager   --circuits amp2 FC comp ldo --k 3 --gnn_embedding_mode sg   --cuda_visible_devices 6 --vllm_model /data/karthik/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct   --apply_chat_template --temperature 0 --max_tokens 8 --batch_size 4 --max_model_len 8192   --out_jsonl filtered_qa_eval_llama3_8b_local.jsonl   --summary_json filtered_qa_eval_llama3_8b_local_summary.json

CUDA_VISIBLE_DEVICES=6 /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py   --eval_vllm_local --log_timings --enforce_eager   --circuits ldo --k 1 --gnn_embedding_mode sg   --cuda_visible_devices 6 --vllm_model /data/karthik/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct   --apply_chat_template --temperature 0 --max_tokens 8 --batch_size 4 --max_model_len 8192   --out_jsonl filtered_qa_eval_llama3_8b_local.jsonl   --summary_json filtered_qa_eval_llama3_8b_local_summary.json


CUDA_VISIBLE_DEVICES=6 /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py \
  --eval_vllm_local \
  --log_timings \
  --enforce_eager \
  --circuits ldo \
  --k 1 \
  --gnn_embedding_mode sg \
  --cuda_visible_devices 6 \
  --vllm_model models--Qwen--Qwen2-7B-Instruct \
  --apply_chat_template \
  --temperature 0 \
  --max_tokens 8 \
  --batch_size 4 \
  --out_jsonl filtered_qa_eval_qwen2_7b.jsonl \
  --summary_json filtered_qa_eval_qwen2_7b_summary.json

  #################
  Models:

llama3_8b
llama3-70b
qwen2-30b
qwen2_7b
qwen2.5_32b
gpt-4.1
gpt-5.1
deepseek_qwen14b
deepseek_qwen32b
deepseek--llama-70b
phi4_reasoning

# Requested local models
# qwen2-30b is wired as an alias to the locally cached Qwen2.5-32B-Instruct checkpoint.
CUDA_VISIBLE_DEVICES=6 /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py \
  --eval_vllm_local \
  --log_timings \
  --enforce_eager \
  --circuits amp2 FC comp \
  --k 2 \
  --gnn_embedding_mode sg \
  --cuda_visible_devices 6 \
  --vllm_model qwen2-30b \
  --apply_chat_template \
  --temperature 0 \
  --max_tokens 8 \
  --batch_size 4 \
  --tensor_parallel_size 1

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py \
  --eval_vllm_local \
  --log_timings \
  --enforce_eager \
  --circuits amp2 FC comp \
  --k 2 \
  --gnn_embedding_mode sg \
  --cuda_visible_devices 0,1,2,3 \
  --vllm_model llama3-70b \
  --apply_chat_template \
  --temperature 0 \
  --max_tokens 8 \
  --batch_size 4 \
  --tensor_parallel_size 4

CUDA_VISIBLE_DEVICES=6 /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py \
  --eval_vllm_local \
  --log_timings \
  --enforce_eager \
  --circuits amp2 FC comp \
  --k 3 \
  --gnn_embedding_mode sg \
  --cuda_visible_devices 6 \
  --vllm_model llama3_8b \
  --apply_chat_template \
  --temperature 0 \
  --max_tokens 8 \
  --batch_size 4

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py \
  --eval_vllm_local \
  --log_timings \
  --enforce_eager \
  --circuits amp2 FC comp \
  --k 3 \
  --gnn_embedding_mode sg \
  --cuda_visible_devices 0,1,2,3 \
  --vllm_model llama3-70b \
  --apply_chat_template \
  --temperature 0 \
  --max_tokens 8 \
  --batch_size 4
  --tensor_parallel_size 4

CUDA_VISIBLE_DEVICES=4 /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py \
  --eval_vllm_local \
  --log_timings \
  --enforce_eager \
  --circuits amp2 FC comp \
  --k 3 \
  --gnn_embedding_mode sg \
  --cuda_visible_devices 4 \
  --vllm_model deepseek--llama-70b \
  --apply_chat_template \
  --temperature 0 \
  --max_tokens 8 \
  --batch_size 4


models = [gpt-4.1, gpt-5.1]

# Requested OpenAI models
OPENAI_API_KEY="$OPENAI_API_KEY" /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py \
  --eval_vllm \
  --log_timings \
  --circuits amp2 FC comp \
  --k 2 \
  --gnn_embedding_mode sg \
  --gnn_checkpoint /home/karthik/sim_clean/AMS-opt/AMS-opt/checkpoints_sg_vs_skg/best_model.pt \
  --embeddings_json /home/karthik/sim_clean/AMS-opt/AMS-opt/umap_results_full/gnn_embeddings_sg_sg_vs_skg.json \
  --vllm_base_url https://api.openai.com \
  --vllm_model gpt-4.1 \
  --temperature 0 \
  --max_tokens 8 \
  --max_completion_tokens 64 \
  --timeout_s 120

OPENAI_API_KEY="$OPENAI_API_KEY" /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py \
  --eval_vllm \
  --log_timings \
  --circuits amp2 FC comp \
  --k 2 \
  --gnn_embedding_mode sg \
  --gnn_checkpoint /home/karthik/sim_clean/AMS-opt/AMS-opt/checkpoints_sg_vs_skg/best_model.pt \
  --embeddings_json /home/karthik/sim_clean/AMS-opt/AMS-opt/umap_results_full/gnn_embeddings_sg_sg_vs_skg.json \
  --vllm_base_url https://api.openai.com \
  --vllm_model gpt-5.1 \
  --temperature 0 \
  --max_tokens 8 \
  --max_completion_tokens 64 \
  --timeout_s 120

OPENAI_API_KEY="$OPENAI_API_KEY" /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py \
  --eval_vllm \
  --log_timings \
  --circuits amp2 FC comp ldo \
  --k 3 \
  --gnn_embedding_mode sg \
  --vllm_base_url https://api.openai.com \
  --vllm_model gpt-5 \
  --temperature 0 \
  --max_tokens 8 \
  --out_jsonl filtered_qa_eval_gpt4o_mini.jsonl \
  --summary_json filtered_qa_eval_gpt4o_mini_summary.json
  --gnn_checkpoint /home/karthik/sim_clean/AMS-opt/AMS-opt/checkpoints_sg_vs_skg/best_model.pt  
  --embeddings_json /home/karthik/sim_clean/AMS-opt/AMS-opt/umap_results_full/gnn_embeddings_sg_sg_vs_skg.json


# QA Accuracy for 5.1 model
OPENAI_API_KEY="$OPENAI_API_KEY" /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py   --eval_vllm   --log_timings   --circuits amp2  --k 1   --kg_max_chars 12000   --gnn_embedding_mode sg   --vllm_base_url https://api.openai.com   --vllm_model gpt-5.1  --timeout_s 60   --max_completion_tokens 8   --out_jsonl filtered_qa_eval_gpt4o_mini.jsonl   --summary_json filtered_qa_eval_gpt4o_mini_summary.json --gnn_checkpoint /home/karthik/sim_clean/AMS-opt/AMS-opt/checkpoints_sg_vs_skg/best_model.pt   --embeddings_json /home/karthik/sim_clean/AMS-opt/AMS-opt/umap_results_full/

OPENAI_API_KEY="$OPENAI_API_KEY" /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py \
  --eval_vllm \
  --log_timings \
  --circuits FC comp \
  --k 1 \
  --gnn_embedding_mode sg \
  --gnn_checkpoint /home/karthik/sim_clean/AMS-opt/AMS-opt/checkpoints_sg_vs_skg/best_model.pt \
  --embeddings_json /home/karthik/sim_clean/AMS-opt/AMS-opt/umap_results_full/gnn_embeddings_sg_sg_vs_skg.json \
  --vllm_base_url https://api.openai.com \
  --vllm_model gpt-5.1 \
  --temperature 1 \
  --max_tokens 8 \
  --max_completion_tokens 64 \
  --timeout_s 120gnn_embeddings_sg_sg_vs_skg.json

# QA accuracy non-gpt models
CUDA_VISIBLE_DEVICES=1,2,3,4 /home/karthik/.conda/envs/analog-rep/bin/python QA_Task/eval_filtered_qa.py   --eval_vllm_local   --log_timings   --enforce_eager   --circuits ldo   --k 2   --gnn_embedding_mode sg   --gnn_checkpoint /home/karthik/sim_clean/AMS-opt/AMS-opt/checkpoints_sg_vs_skg/best_model.pt   --embeddings_json /home/karthik/sim_clean/AMS-opt/AMS-opt/umap_results_full/gnn_embeddings_sg_sg_vs_skg.json   --cuda_visible_devices 1,2,3,4   --vllm_model llama3-70b   --apply_chat_template   --temperature 1   --max_tokens 8   --batch_size 4 --gnn_embedding_mode sg --tensor_parallel_size 2 --kg_max_chars 10000