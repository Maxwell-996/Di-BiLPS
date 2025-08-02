#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

"""Train the custom GINOAutoencoderKL on PDE field pairs."""

import argparse
import contextlib
import gc
import logging
import math
import os
import shutil
from pathlib import Path
import torch.nn as nn
import sys
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_parent_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, parent_parent_parent_dir)
from AutoencoderKL import GINOAutoencoderKL
if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a AutoencoderKL training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the VAE model to train, leave as None to use standard VAE model configuration.",
    )
    parser.add_argument(
        "--discriminator_config_name_or_path",
        type=str,
        default=None,
        help="The config of the VAE model to train, leave as None to use standard VAE model configuration.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="autoencoderkl-model/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--in_channels", type=int, default=3, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4.5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--disc_learning_rate",
        type=float,
        default=4.5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--disc_lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help="A set of paths to the image be evaluated every `--validation_steps` and logged to `--report_to`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=20,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_autoencoderkl",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--rec_loss",
        type=str,
        default="l2",
        help="The loss function for VAE reconstruction loss.",
    )
    parser.add_argument(
        "--kl_scale",
        type=float,
        default=1e-6,
        help="Scaling factor for the Kullback-Leibler divergence penalty term.",
    )
    parser.add_argument(
        "--perceptual_scale",
        type=float,
        default=0.5,
        help="Scaling factor for the LPIPS metric",
    )
    parser.add_argument(
        "--disc_start",
        type=int,
        default=50001,
        help="Start for the discriminator",
    )
    parser.add_argument(
        "--disc_factor",
        type=float,
        default=1.0,
        help="Scaling factor for the discriminator",
    )
    parser.add_argument(
        "--disc_scale",
        type=float,
        default=1.0,
        help="Scaling factor for the discriminator",
    )
    parser.add_argument(
        "--disc_loss",
        type=str,
        default="hinge",
        help="Loss function for the discriminator",
    )
    parser.add_argument(
        "--decoder_only",
        action="store_true",
        help="Only train the VAE decoder.",
    )
    parser.add_argument(
        "--use_irr_o",
        action="store_true",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.pretrained_model_name_or_path is not None and args.model_config_name_or_path is not None:
        raise ValueError("Cannot specify both `--pretrained_model_name_or_path` and `--model_config_name_or_path`")

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the diffusion model."
        )
    

    return args


def make_train_dataset(args, accelerator):
    train_dataset_path = "./{}/detail/".format(args.dataset_name)
    val_dataset_path = "/home/ubuntu/PDE_data/testing/{}/detail/".format(args.dataset_name)
    
    dataset = load_dataset(
    path=f"./data_gen/{args.dataset_name}_data_gen.py",   # 
    data_dir=train_dataset_path,    trust_remote_code=True  ,writer_batch_size=1000  # 
    )       
    
    val_dataset = load_dataset(
    path=f"./data_gen/{args.dataset_name}_data_gen.py",   # 
    data_dir=val_dataset_path,    trust_remote_code=True  ,writer_batch_size=1000  #
    ) 
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"]
        val_dataset = val_dataset["train"]
    return train_dataset,val_dataset


def collate_fn(examples):
    pixel_values = torch.stack([torch.tensor(example["UA"]) for example in examples])
   
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    return {"U": pixel_values}


def main(args):
    from datetime import datetime
    import os
    time_str = datetime.now().strftime("%m_%d_%H_%M_%S")
    time_str = f"{args.dataset_name}_"+time_str
    args.output_dir = os.path.join(args.output_dir,time_str)
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load AutoencoderKL
    if args.use_irr_o:
        aemodel = GINOAutoencoderKL
    else:
        aemodel = AutoencoderKL
    
    if args.pretrained_model_name_or_path is None and args.model_config_name_or_path is None:
        config = aemodel.load_config("stabilityai/sd-vae-ft-mse")
        vae = aemodel.from_config(config)
    elif args.pretrained_model_name_or_path is not None:
        vae = aemodel.from_pretrained(args.pretrained_model_name_or_path)
    else:
        config = aemodel.load_config(args.model_config_name_or_path)
        vae = aemodel.from_config(config)
    if args.use_ema:
        ema_vae = EMAModel(vae.parameters(), model_cls=aemodel, model_config=vae.config)
    perceptual_loss = lpips.LPIPS(net="vgg").eval()

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model


    vae.requires_grad_(True)
    if args.decoder_only:
        vae.encoder.requires_grad_(False)
        if getattr(vae, "quant_conv", None):
            vae.quant_conv.requires_grad_(False)
    vae.train()


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            vae.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(vae).dtype != torch.float32:
        raise ValueError(f"VAE loaded as datatype {unwrap_model(vae).dtype}. {low_precision_error_string}")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )


    optimizer_class = torch.optim.AdamW

    params_to_optimize = filter(lambda p: p.requires_grad, vae.parameters())
    
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    train_dataset,val_dataset = make_train_dataset(args, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    (
        vae,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        vae, optimizer, train_dataloader,val_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move VAE, perceptual loss and discriminator to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    perceptual_loss.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        ema_vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        vae.train()
        for step, batch in enumerate(train_dataloader):
            targets = batch["U"].to(dtype=weight_dtype)
            bs = targets.shape[0]
            posterior = accelerator.unwrap_model(vae).encode(targets).latent_dist
            latents = posterior.sample()
            reconstructions = accelerator.unwrap_model(vae).decode(latents,expand_time=1).sample
            timing_version = False
            
            with accelerator.accumulate(vae):
                # reconstruction loss. Pixel level differences between input vs output
                if  args.dataset_name == "darcy":
                    bce_fn = nn.BCEWithLogitsLoss(reduction='none')
                    rec_loss = F.mse_loss(reconstructions[:,:1,...].float(), targets[:,:1,...].float(), reduction="none")
                    mask = (targets[:,1:,...] > 0).float()
                    pred = reconstructions[:,1:,...] 
                    bce_loss = bce_fn(pred,mask)
                    rec_loss = 0.1*bce_loss + rec_loss
                    pred_mask = (torch.sigmoid(pred) > 0.5).float()
                    acc  = ( (pred_mask == mask).float().sum()  ) / (mask.numel())
                elif args.rec_loss == "l2":
                    rec_loss = F.mse_loss(reconstructions.float(), targets.float(), reduction="none")
                elif args.rec_loss == "l1":
                    rec_loss = F.l1_loss(reconstructions.float(), targets.float(), reduction="none")
                else:
                    raise ValueError(f"Invalid reconstruction loss type: {args.rec_loss}")
                # perceptual loss. The high level feature mean squared error loss
                if timing_version:
                    with torch.no_grad():
                        B,C,H,W = reconstructions.shape
                        p_loss = []
                    
                        reconstructions = reconstructions.view(4,B//4,C,H,W)
                        targets = targets.view(4,B//4,C,H,W)
                        for i in range(4):
                            p_loss.append(perceptual_loss(reconstructions[i].expand(B//4,3,H,W),targets[i].expand(B//4,3,H,W))) 
                    p_loss = torch.cat(p_loss,dim = 0)
                else:
                    with torch.no_grad():
                        B,C,H,W = reconstructions.shape
                    
                        if reconstructions.shape[1] == 2:
                            if args.dataset_name == "darcy":
                                p_loss = perceptual_loss(reconstructions[:,:1,...],targets[:,:1,...])
                                
                            else:
                                p_loss_U = perceptual_loss(reconstructions[:,:1,...],targets[:,:1,...]) 
                                p_loss_A = perceptual_loss(reconstructions[:,1:,...],targets[:,1:,...])
                                p_loss = torch.cat([p_loss_U,p_loss_A],dim = 1)
                        else:
                            p_loss = perceptual_loss(reconstructions,targets)
                    
                
                nll_loss = 20*rec_loss + args.perceptual_scale * p_loss
                nll_loss = torch.sum(nll_loss) / bs 

                kl_loss = posterior.kl()
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]


                loss = nll_loss + args.kl_scale * kl_loss 

                logs = {
                    "rec_loss": rec_loss.detach().mean().item(),
                }
                if args.dataset_name == "darcy":
                    L2_err =  torch.norm(targets[0,:1,...] - reconstructions[0,:1,...], 2) / torch.norm(targets[0,:1,...], 2)
                    logs.update({"a_L2_err_u": L2_err.detach().item()})
                    logs.update({"acc": acc.detach().item()})
                    logs.update({"bce_loss": bce_loss.mean().detach().item()})
                else:
                    L2_err =  torch.norm(targets[0,:1,...] - reconstructions[0,:1,...], 2) / torch.norm(targets[0,:1,...], 2)
                    logs.update({"a_L2_err_u": L2_err.detach().item()})
                    L2_err =  torch.norm(targets[0,1:,...] - reconstructions[0,1:,...], 2) / torch.norm(targets[0,1:,...], 2)
                    logs.update({"a_L2_err_a": L2_err.detach().item()})
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = vae.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
        
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if args.use_ema:
                    ema_vae.step(vae.parameters())

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 or global_step == 2:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.unwrap_model(vae).save_pretrained(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step == 1 or global_step % args.validation_steps == 0:
                        if args.use_ema:
                            ema_vae.store(vae.parameters())
                            ema_vae.copy_to(vae.parameters())
                        if args.use_ema:
                            ema_vae.restore(vae.parameters())
            if accelerator.is_main_process:  # or use accelerate.is_main_process if using accelerate           
                log_to_csv_pandas(logs,filename=os.path.join(args.output_dir,"training_log.csv"))
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        vae.eval()   
        with torch.no_grad():
            test_L2_error = []
            acc_arr = []
            for step, batch in enumerate(val_dataloader):
                # Convert images to latent space and reconstruct from them
                targets = batch["U"].to(dtype=weight_dtype)
                bs = targets.shape[0]
                posterior = accelerator.unwrap_model(vae).encode(targets).latent_dist
                latents = posterior.sample()
                reconstructions = accelerator.unwrap_model(vae).decode(latents,expand_time=1).sample
                if len(targets.shape) > 4:
                    targets = targets.permute(0,1,4,2,3)
                    if args.in_channels != 1:
                        ndims = len(targets.shape)
                        targets = targets.flatten(0,ndims-3).unsqueeze(1)
                    if  reconstructions.shape[1] != 1 :
                        reconstructions = reconstructions.flatten(0,1).unsqueeze(1)   
                if  args.dataset_name == "darcy":
                    bce_fn = nn.BCEWithLogitsLoss(reduction='none')
                    rec_loss = F.mse_loss(reconstructions[:,:1,...].float(), targets[:,:1,...].float(), reduction="none")
                    mask = (targets[:,1:,...] > 0).float()
                    pred = reconstructions[:,1:,...] 
                    bce_loss = bce_fn(pred,mask)
                    rec_loss = bce_loss + rec_loss
                    pred_mask = (torch.sigmoid(pred) > 0.5).float()
                    acc  = ( (pred_mask == mask).float().sum()  ) / (mask.numel())
                elif args.rec_loss == "l2":
                    rec_loss = F.mse_loss(reconstructions.float(), targets.float(), reduction="none")
                elif args.rec_loss == "l1":
                    rec_loss = F.l1_loss(reconstructions.float(), targets.float(), reduction="none")
                else:
                    raise ValueError(f"Invalid reconstruction loss type: {args.rec_loss}")


                logs = {
                    "test_metric": 0,
                    "rec_loss": rec_loss.detach().sum().item(),
                }
                if args.dataset_name == "darcy":
                    L2_err =  torch.norm(targets[0,:1,...] - reconstructions[0,:1,...], 2) / torch.norm(targets[0,:1,...], 2)
                    logs.update({"a_L2_err": L2_err.detach().item()})
                    logs.update({"acc": acc.detach().item()})
                    acc_arr.append(acc)
                else:
                    L2_err =  torch.norm(targets[0,:1,...] - reconstructions[0,:1,...], 2) / torch.norm(targets[0,:1,...], 2)
                    logs.update({"a_L2_err_u": L2_err.detach().item()})
                    L2_err =  torch.norm(targets[0,1:,...] - reconstructions[0,1:,...], 2) / torch.norm(targets[0,1:,...], 2)
                    logs.update({"a_L2_err_a": L2_err.detach().item()})
                test_L2_error.append(L2_err.item())
            if len(acc_arr) > 0:
                print("test acc: ",  sum(acc_arr) / len(acc_arr))    
            print("test a_L2_err: ",  sum(test_L2_error) / len(test_L2_error))
 
    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        if args.use_ema:
            ema_vae.copy_to(vae.parameters())
        vae.save_pretrained(args.output_dir)
        # Run a final round of validation.
        image_logs = None


    accelerator.end_training()
    
    
import pandas as pd
def log_to_csv_pandas(logs, filename='training_log.csv'):
    #        DataFrame
    df_new = pd.DataFrame([logs])
    #        ，    
    if not os.path.exists(filename):
        df_new.to_csv(filename, index=False)
    else:
        #       
        df_existing = pd.read_csv(filename)

        #      
        df_combined = pd.concat([df_existing, df_new], ignore_index=True).fillna("")

        # 
        df_combined.to_csv(filename, index=False)        
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
