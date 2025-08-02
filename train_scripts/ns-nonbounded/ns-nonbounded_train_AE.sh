CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch training/train_AE.py \
  --pretrained_model_name_or_path ./ns-nonbounded/vae \
  --dataset_name ns-nonbounded \
  --resolution=128 \
  --train_batch_size=24 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs 200 \
  --in_channels 2  \
  --output_dir /data/ckpt_management_dir/my_vae_model \
  --use_irr_o
