import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from transformers import CLIPVisionConfig, CLIPModel,CLIPVisionModel
import json
import numpy as np
from datasets import load_dataset
import argparse
import shutil
import math
EPOCH = 200
from transformers import PreTrainedModel, PretrainedConfig
from clip_model import CLIP,MyConfig

"""Train the CLIP-style condition encoder used by the diffusion model."""


def get_args():
    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument('--checkpoints_dir', type=str, default='/home/ubuntu/PDE_data/ckpt_manage_dir/ckpt_management_dir/',
                        help='Directory to save model checkpoints')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--dataset_name', type=str, default='ns-nonbounded',
                        help='Name of the dataset to use')
    parser.add_argument('--device', type=int, default='1',
                        help='Device to use for training (e.g., "cuda", "cuda:1", or "cpu")')
    parser.add_argument('--max_sampled', type=int, default=16)
    parser.add_argument('--config_path', type=str,default="./configs/")
    args = parser.parse_args()
    return args

    
def collate_fn(examples):
    """Build contrastive pairs from dataset samples.

    Datasets with `UA` provide paired channels directly. Older temporal
    datasets without `UA` are sampled into current/target pairs with timesteps.
    """
    
    if examples[0].get("UA") is None:
        max_sampled = args.max_sampled
        num_tstep = len(examples[0]["U"])  
        
        timesteps = []
        ticks =  np.random.randint(0, num_tstep,size=len(examples)) 
        index_list = []
        for id in range(len(examples)):
            tick = ticks[id]
            if  max_sampled < num_tstep:
                index = np.random.choice(num_tstep, size=max_sampled, replace=False, p=None)
                index_list.append(index)
                timesteps = timesteps + [i-tick+num_tstep for i in index]
            else:
                timesteps = timesteps + [i-tick+num_tstep for i in range(num_tstep)]
            
        A = [torch.tensor(example["U"])[ticks[i]:ticks[i]+1] for i,example in enumerate(examples)]
        A = torch.stack(A,dim=0)     
        
        if examples[0].get("cond") is not None:
            cond_nums = min(num_tstep,max_sampled)
            cond = [torch.tensor([example["cond"]]*cond_nums) for i,example in enumerate(examples)]
            cond = torch.stack(cond,dim=0).reshape(-1)
        else:    
            cond = None   
            
              
        if  max_sampled < num_tstep:
            U = torch.stack([torch.tensor(example["U"])[index_list[i],...] for i,example in enumerate(examples)])
        else:
            U = torch.stack([torch.tensor(example["U"]) for i,example in  enumerate(examples)])
        
        timesteps = torch.tensor(timesteps,dtype=torch.float32)
        
        
        return {
            "cond": cond,
            "U": U,
            "A": A,
            "timesteps": timesteps
        }
    else:
        U = torch.stack([torch.tensor(example["UA"])[:1,...] for example in examples])
        A = torch.stack([torch.tensor(example["UA"])[1:,...] for example in examples])
        
        data = {
            "U": U,
            "A": A,
        }    
        if examples[0].get("cond") is not None:
            cond = torch.stack([torch.tensor(example["cond"])[0] for example in examples])
            data.update({
                "cond": cond
            })
        return data
    
if __name__ == "__main__":
    args = get_args()
    checkpoints_dir = os.path.join(args.checkpoints_dir,args.dataset_name)
    dataset_name = args.dataset_name
    BATCH_SIZE = args.batch_size
    ckpt_path = args.ckpt_path
    device  =  f"cuda:{args.device}"
    config_vit = MyConfig(args.config_path)
    model = CLIP(config=config_vit)
    train_dataset_path = "./{}/detail/".format(dataset_name)
    val_dataset_path = "/home/ubuntu/PDE_data/testing/{}/detail/".format(dataset_name)
    train_dataset = load_dataset(
    path=f"./data_gen/{dataset_name}_data_gen.py",   #         
    data_dir=train_dataset_path,    trust_remote_code=True  , writer_batch_size=1000 ,
    )["train"] 
    val_dataset = load_dataset(
    path=f"./data_gen/{dataset_name}_data_gen.py",   #         
    data_dir=val_dataset_path,    trust_remote_code=True, writer_batch_size=1000,
    )["train"]
    if ckpt_path is not None:
        print("Loading checkpoint from {}".format(ckpt_path))
        model = model.from_pretrained(ckpt_path)
    model.to(device)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=16,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=16,
    )
    

    
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)

    # checkpoint    
    saved_avg_val_loss = 1e9
    from datetime import datetime
    now = datetime.now()
    time_str = now.strftime("%m_%d_%H_%M_%S")
    best_checkpoints = []
    #        
    for epoch in range(EPOCH):
        print(f"\nEpoch {epoch + 1}/{EPOCH}")
        model.train()
        total_epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc="Training", leave=False)

        for batch in progress_bar:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            logits_U,logits_A = model(batch)
            ground_truth = torch.arange(logits_U.shape[0], dtype=torch.long, device=device)
            total_loss = (loss_img(logits_U, ground_truth) + loss_txt(logits_A, ground_truth)) / 2

            total_loss.backward()
            optimizer.step()

            total_epoch_loss += total_loss.item()
            progress_bar.set_postfix(loss=total_loss.item())

        avg_train_loss = total_epoch_loss / len(train_dataloader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        #      
        model.eval()
        total_val_loss = 0.0
        total_correct = 0
        total_samples = 0
        progress_bar = tqdm(val_dataloader, desc="Testing", leave=False)
        with torch.no_grad():
            for val_batch in progress_bar:
                val_batch = {k: v.to(device)  for k, v in val_batch.items()}
                logits_U,logits_A = model(val_batch)
                #   
                ground_truth = torch.arange(logits_U.shape[0], dtype=torch.long, device=device)
                val_loss = (loss_img(logits_U, ground_truth) + loss_txt(logits_A, ground_truth)) / 2
                total_val_loss += val_loss.item()

                #      （     ）
                pred = torch.argmax(logits_U, dim=1)
                correct = (pred == ground_truth).sum().item()
                total_correct += correct
                total_samples += logits_U.shape[0]

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = total_correct / total_samples
        
        
        print(f"Val Loss: {avg_val_loss:.4f} | Matching Accuracy: {val_accuracy:.4f}")
        
        os.makedirs(checkpoints_dir, exist_ok=True)
        name_id =  f"{dataset_name}_{time_str}/{epoch+1:03d}/"
        path = os.path.join(checkpoints_dir,name_id)
        if  epoch % 10 ==0 or epoch == EPOCH-1:
            model.save_pretrained(save_directory=path)
