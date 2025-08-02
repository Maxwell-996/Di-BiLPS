 


python inference_data.py \
  --pretrained_model_name_or_path=./helmholtz/diff \
  --pretrained_vae_model_name_or_path=./helmholtz/vae \
  --pretrained_clip_model_name_or_path=./helmholtz/rep_ali\
  --dataset_name=helmholtz \
  --pretrained_sheduler_model_name_or_path=./sampler \
  --use_irr_data \
  --obs_guide_a_weight 0.1 \
  --obs_guide_u_weight 0 \
  --pde_guideweight 1 \
  --inference_steps 200 \
