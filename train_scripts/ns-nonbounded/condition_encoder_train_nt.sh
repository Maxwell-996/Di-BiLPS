python train_img_encoder.py \
  --ckpt_path  ./ns-nonbounded/rep_ali \
  --checkpoints_dir /data/ckpt_management_dir/my_clip_vision_model/ \
  --batch_size 32 \
  --dataset_name ns-nonbounded \
  --device 1 \
  --config_path ./configs/ns-nonbounded/
