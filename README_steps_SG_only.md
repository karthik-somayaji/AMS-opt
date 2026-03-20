# Training
/home/karthik/miniconda3/envs/analog-rep/bin/python -u gnn_training/scripts/train.py \
  --config gnn_training/config/full_training_diff_amps_comparators_LDO_opamp_all.yaml \
  --data.data_dir /home/karthik/sim_clean/AMS-opt/netlists

# Visualization
/home/karthik/miniconda3/envs/analog-rep/bin/python -u gnn_training/scripts/analyze_sg_skg_embeddings.py \
  --checkpoint checkpoints_sg_vs_skg/best_model.pt \
  --netlists_root /home/karthik/sim_clean/AMS-opt/netlists \
  --out_dir /home/karthik/sim_clean/AMS-opt/umap_results_full/sg_skg_analysis \
  --export_llmbo_json

python3 scripts/visualize_embeddings_umap.py   --config gnn_training/config/full_training_diff_amps_comparators_LDO_opamp_all.yaml   --device cpu

# Optimization
cd /home/karthik/sim_clean/AMS-opt && /home/karthik/miniconda3/envs/analog-rep/bin/python LLMBO/llmbo.py   --history 1 --related_mode bottomk --related_k 3 --target_id FC   --gnn_embedding_mode sg

# QA Dataset Generation
cd /home/karthik/sim_clean/AMS-opt
/home/karthik/miniconda3/envs/analog-rep/bin/python filter_with_llmbo.py \
  --circuit FC --paragraph_k -1 --repeats 3 --n_itr 5 --n_proposal_llm 1 --n_proposal_bo 0 \
  --min_delta 0.0 --llm_temperature 0.0