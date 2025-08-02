
python inference_data.py \
  --pretrained_model_name_or_path=./darcy/diff \
  --pretrained_vae_model_name_or_path=./darcy/vae \
  --pretrained_clip_model_name_or_path=./darcy/rep_ali\
  --dataset_name=darcy \
  --pretrained_sheduler_model_name_or_path=./sampler \
  --use_irr_data \
  --obs_guide_a_weight 20 \
  --obs_guide_u_weight 0 \
  --pde_guideweight 1 \
  --inference_steps 200
