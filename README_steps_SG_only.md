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