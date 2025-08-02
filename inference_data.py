import torch
import numpy as np
from  diffusers.pipelines.stable_diffusion.pipeline_PDE_guided_sd import StableDiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, UNet2DConditionModel,DDIMScheduler
from transformers import CLIPVisionModel
from datasets import  load_dataset
from torch.utils.data import Dataset, DataLoader
import os,sys
from torchvision.utils import save_image
import torchvision.transforms as transforms
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from train_img_encoder import CLIP
from clip_model import MyConfig
from AutoencoderKL import GINOAutoencoderKL
import time

"""Run PDE-guided sampling with trained VAE, CLIP encoder, scheduler, and UNet."""

def get_args():
    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--pretrained_sheduler_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--pretrained_clip_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--dataset_name', type=str, default='ns-nonbounded',
                        help='Name of the dataset to use')
    parser.add_argument('--max_sample_len', type=int, default=64,
                        help='Maximum number of sparse observations used per sample (for irregular guidance).')
    parser.add_argument('--inference_steps', type=int, default=200,
                        help='Number of denoising steps during sampling.')
    parser.add_argument('--obs_guide_a_weight', type=float, default=0,
                        help='Guidance weight for observed A-field consistency.')
    parser.add_argument('--obs_guide_u_weight', type=float, default=0,
                        help='Guidance weight for observed U-field consistency.')
    parser.add_argument('--pde_guideweight', type=float, default=0,
                        help='Guidance weight for PDE residual constraint.')
    parser.add_argument(
        "--use_irr_data",
        action="store_true",
    )
    parser.add_argument(
        "--inverse",
        action="store_true",
    )
    args = parser.parse_args()
    return args

h_data_max = 0.022490255461219907
h_data_min = -0.02734944077477767
h_data_std = 0.004270045990050305
h_data_mean = 9.46510862536118e-06
h_data_a_max = 2.024150613827505
h_data_a_min = -1.8881286627715816
h_data_a_std = 0.28432797574133245
h_data_a_mean = -2.5465973392311507e-06


def batch_random_mask_vec(data, batch_size, k, grid_size, seed=0, device=None):
    """
    Sample observation masks and return both dense masks and sparse query data.
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    total = grid_size * grid_size

    #      0 mask
    masks = torch.zeros((batch_size, total), dtype=torch.bool, device=device)
    indices = []
    values = []
    data = data.view(batch_size,-1,1)
    for i in range(batch_size):
        idx = torch.randperm(total, device=device)[:k]
        masks[i, idx] = True
        id_row = idx//grid_size
        id_col = idx%grid_size
        ind = torch.stack([id_row,id_col],dim=-1) 
        indices.append(ind)
        values.append(data[i,idx])
    indices = torch.stack(indices)
    values = torch.stack(values)
    position  = indices / grid_size
    # size (batchsize,k)
    dict_data = {
        "indices": indices,
        "values":values,
        "position":position
    }
    return masks.view(batch_size,1,grid_size, grid_size), dict_data

def recover_dict_data_from_mask(data, mask):
    """
         mask   data    dict_data（indices, values, position）

      ：
        data: Tensor of shape (batch_size, 1, grid_size, grid_size)
        mask: BoolTensor of shape (batch_size, 1, grid_size, grid_size)

     ：
        dict_data:
            - indices: (batch_size, k, 2)         
            - values:  (batch_size, k, 1)        
            - position: (batch_size, k, 2)       [0, 1)
    """
    batch_size, _, grid_size, _ = mask.shape
    total = grid_size * grid_size
    data = data.view(batch_size, -1, 1)  # shape: (B, total, 1)
    mask_flat = mask.view(batch_size, -1)  # shape: (B, total)

    indices_list = []
    values_list = []
    for i in range(batch_size):
        idx = torch.nonzero(mask_flat[i], as_tuple=False).squeeze(1)  # shape: (k,)
        val = data[i, idx]  # shape: (k, 1)

        id_row = idx // grid_size
        id_col = idx % grid_size
        ind = torch.stack([id_row, id_col], dim=-1)  # (k, 2)

        indices_list.append(ind)
        values_list.append(val)

    indices = torch.stack(indices_list)  # (B, k, 2)
    values = torch.stack(values_list)    # (B, k, 1)
    position = indices / grid_size       #   [0, 1)

    dict_data = {
        "indices": indices,
        "values": values,
        "position": position
    }
    return dict_data


if __name__ == "__main__":
    args = get_args()
    model_path = args.pretrained_model_name_or_path
    device = "cuda:0"
    dataset_path = f"./test_data/{args.dataset_name}/"
    data_path = os.path.join(dataset_path,"merge_0.npy")
    data = torch.tensor(np.load(data_path)).to(device)
    mask_path = os.path.join(dataset_path,"mask.npy")
    mask_data = torch.tensor(np.load(mask_path)).to(device)
    if args.inverse:
        dict_data = recover_dict_data_from_mask(data[:1,...], mask_data)
    else:
        dict_data = recover_dict_data_from_mask(data[1:,...], mask_data)
    mask = (mask_data,dict_data)
    if args.use_irr_data:
        aemodel = GINOAutoencoderKL
    else:
        aemodel = AutoencoderKL

    text_encoder = CLIP.from_pretrained(args.pretrained_clip_model_name_or_path)
    output_dir = "./vis_output"
    output_dir = os.path.join(output_dir,args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)  
    vae = aemodel.from_pretrained(args.pretrained_vae_model_name_or_path)
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_sheduler_model_name_or_path)
    unet = UNet2DConditionModel.from_pretrained(model_path)
    pipeline = StableDiffusionPipeline(unet=unet,
                                    vae=vae,  #  model.safetensors
                                    scheduler = noise_scheduler,
                                    text_encoder = text_encoder,
                                    safety_checker = None,
                                    feature_extractor = None)
                                                    
    pipeline.to(device)
    torch.cuda.manual_seed(42)  
    data_dict = {
        "U":data[:1,...],
        "A":data[1:,...],
    }
    dataloader = [data_dict]

    import matplotlib.pyplot as plt
    def save_tensor_with_colorbar(tensor, output_path, cmap='viridis'):
        #     
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # [1, H, W] -> [H, W]
        elif tensor.dim() == 3 and tensor.shape[0] == 3:
            raise ValueError("This function is for grayscale tensors, got 3-channel input.")

        #   
        plt.figure(figsize=(5, 5))
        im = plt.imshow(tensor.cpu().numpy(), cmap=cmap)
        plt.colorbar(im)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    for batch_idx, batch in enumerate(dataloader):
        
        # shape: (B, C, H, W)
        if batch.get("A") is None:
            Gt = batch["U"]
            ticks = np.random.randint(1, 11,size = batch["U"].shape[0]) 
            outputs = pipeline(prompt=Gt,ticks = ticks ,height=128,width=128,guidance_scale=0,num_inference_steps = 200).images  # 
        else:
            if args.inverse:
                prompt = batch["U"]
            else:
                prompt = batch["A"]
            loss_weights = (args.obs_guide_a_weight,args.obs_guide_u_weight,args.pde_guideweight)
            start = time.time()
            outputs = pipeline(
                prompt=prompt,
                dataset_name=args.dataset_name,
                is_inverse = args.inverse,
                u_real = batch["U"],
                a_real = batch["A"],
                height=128,width=128,
                guidance_scale=0,
                num_inference_steps = args.inference_steps,
                loss_weights = loss_weights,
                mask = mask
                ).images
            mask = pipeline.mask
            end = time.time()
            print(f"time: {end - start:.4f}")
        with torch.no_grad():
            for i in range(len(outputs)):
                if batch.get("A") is None:
                    tick = ticks[i]
                    u_GT = Gt[i,...].to(outputs.device)
                    u_final = outputs[i,...]
                    u_final[tick] = u_GT[tick]
                elif args.dataset_name == "darcy":
                    u_GT = batch["U"][i,0,...].to(outputs.device)
                    
                    a_GT = (batch["A"][i,0,...] > 0).to(outputs.device)
                    u_GT = ((u_GT+0.9)/115).to(torch.float64)
                    
                    u_final = outputs[i,0,...]
                    a_final = (outputs[i,1,...] >0)
                    
                    u_final = ((u_final+0.9)/115).to(torch.float64)
                    
                    if args.inverse:
                        u_final[mask.squeeze()] = u_GT[mask.squeeze()]
                    else:
                        a_final[mask.squeeze()] = a_GT[mask.squeeze()]
                    
                   
                    acc = torch.sum(a_final==a_GT) / (128*128)
                    print("acc",acc.item())
                    
                elif args.dataset_name == "helmholtz":
                    u_GT = batch["U"][i,0,...].to(outputs.device)
                    a_GT = batch["A"][i,0,...].to(outputs.device)
                    
                    u_GT = (u_GT*(1e-8 + h_data_std) + h_data_mean).to(torch.float64)
                    a_GT = (a_GT*(1e-8 + h_data_a_std) + h_data_a_mean).to(torch.float64)
                    
                    u_final = outputs[i,0,...]
                    a_final = outputs[i,1,...] 
                    
                    u_final = (u_final*(1e-8 + h_data_std) + h_data_mean).to(torch.float64)
                    a_final = (a_final*(1e-8 + h_data_a_std) + h_data_a_mean).to(torch.float64)
                    
                    if args.inverse:
                        u_final[mask.squeeze()] = u_GT[mask.squeeze()]
                    else:
                        a_final[mask.squeeze()] = a_GT[mask.squeeze()]
                    
                    
                    relative_error_a = torch.norm(a_final - a_GT, 2) / torch.norm(a_GT, 2)
                    print("loss_A:",relative_error_a.item()) 
                    acc = relative_error_a
                elif  args.dataset_name == "poisson":
                    u_GT = batch["U"][i,0,...].to(outputs.device)
                    a_GT = batch["A"][i,0,...].to(outputs.device)
                    
                    u_GT = (u_GT/36.5).to(torch.float64)
                    a_GT = (a_GT*2.15).to(torch.float64)
                    
                    u_final = outputs[i,0,...]
                    a_final = outputs[i,1,...] 
                    
                    u_final = (u_final/36.5).to(torch.float64)
                    a_final = (a_final*2.15).to(torch.float64)
                    if args.inverse:
                        u_final[mask.squeeze()] = u_GT[mask.squeeze()]
                    else:
                        a_final[mask.squeeze()] = a_GT[mask.squeeze()]
                    relative_error_a = torch.norm(a_final - a_GT, 2) / torch.norm(a_GT, 2)
                    print("loss_A:",relative_error_a.item()) 
                    acc = relative_error_a    
                else:    
                    u_GT = batch["U"][i,0,...].to(outputs.device).to(torch.float64)
                    a_GT = batch["A"][i,0,...].to(outputs.device).to(torch.float64)
                    u_final = outputs[i,0,...].to(torch.float64)
                    a_final = outputs[i,1,...] .to(torch.float64)
                    #     u_final[mask.squeeze()] = u_GT[mask.squeeze()]
                    #     a_final[mask.squeeze()] = a_GT[mask.squeeze()]
                    relative_error_a = torch.norm(a_final - a_GT, 2) / torch.norm(a_GT, 2)
                    print("loss_A:",relative_error_a.item()) 
                    acc = relative_error_a
                if args.inverse:
                    save_path_A = os.path.join(dataset_path,"inverse_predicted_data_A.npy")  
                    save_path_U = os.path.join(dataset_path,"inverse_predicted_data_U.npy")   
                else:
                    save_path_A = os.path.join(dataset_path,"512_predicted_data_A.npy")  
                    save_path_U = os.path.join(dataset_path,"512_predicted_data_U.npy")    
                np.save(save_path_U,u_final.cpu().numpy())
                np.save(save_path_A,a_final.cpu().numpy())
                relative_error_u = torch.norm(u_final - u_GT, 2) / torch.norm(u_GT, 2)
                print("loss_U:",relative_error_u.item()) 
                comparison = torch.cat([u_GT, u_final,u_GT-u_final], dim=1) 
                if len(comparison.shape)>2:
                    comparison = torch.cat([comparison[j] for j in range(11)],dim = -1)
                save_tensor_with_colorbar(comparison, os.path.join(output_dir, f"compare_{batch_idx}_{i}_loss_{relative_error_u.item():.3f}.png"))
                comparison = torch.cat([a_GT, a_final], dim=1) 
                save_tensor_with_colorbar(comparison, os.path.join(output_dir, f"compare_{batch_idx}_{i}_acc_{acc.item():.3f}.png"))
                
                
