python train_img_encoder.py \
  --ckpt_path  ./poisson/rep_ali \
  --checkpoints_dir /data/ckpt_management_dir/my_clip_vision_model/ \
  --batch_size 32 \
  --dataset_name poisson \
  --device 0 \
  --config_path ./configs/poisson/
