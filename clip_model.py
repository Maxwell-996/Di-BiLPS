
from transformers import CLIPVisionConfig, CLIPModel,CLIPVisionModel
import json
import numpy as np
import os
from transformers import PreTrainedModel, PretrainedConfig
from timestepEmbedder import TimestepEmbedder, TimestepEmbedding, Timesteps
import torch
import torch.nn as nn
from GINO import GINO_Encoder_wrapper


class MyConfig(PretrainedConfig):
    """Load paired CLIP vision configs for A-to-U and U-to-A conditioning."""

    def __init__(self,path= None,**kwargs):
        super().__init__(**kwargs)
        if path is not None:
            path_A = os.path.join(path,"clip_vit_A.json")
            with open(path_A, "r") as f:
                Aconfig_dict = json.load(f)
            path_U = os.path.join(path,"clip_vit_U.json")
            with open(path_U, "r") as f:
                Uconfig_dict = json.load(f)
            self.configA = CLIPVisionConfig.from_dict(Aconfig_dict)
            self.configU = CLIPVisionConfig.from_dict(Uconfig_dict)
        
class TinyBlock(nn.Module):
    """
    Tiny Autoencoder block used in [`AutoencoderTiny`]. It is a mini residual module consisting of plain conv + ReLU
    blocks.

    Args:
        in_channels (`int`): The number of input channels.
        out_channels (`int`): The number of output channels.
        act_fn (`str`):
            ` The activation function to use. Supported values are `"swish"`, `"mish"`, `"gelu"`, and `"relu"`.

    Returns:
        `torch.Tensor`: A tensor with the same shape as the input tensor, but with the number of channels equal to
        `out_channels`.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        act_fn = nn.SiLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.fuse = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.conv(x) + self.skip(x))        
        
class CLIP(PreTrainedModel):
    """Project PDE fields into CLIP-style embeddings for diffusion conditioning."""

    config_class = MyConfig
    def __init__(self,config=None):
        super().__init__(config)
        
            
        config_A = config.configA
        config_U = config.configU
        if isinstance(config_A, dict):
            config_A = CLIPVisionConfig.from_dict(config_A)
            config_U = CLIPVisionConfig.from_dict(config_U)
            
        self.irr_input = config_A.irr_input
        self.use_GINO = config_A.use_GINO
        self.projection_channels = config_A.num_channels
        self.cond_num = config_A.cond_num if config_A.use_cond else None
        self.obs_num = 500
        if self.irr_input and self.use_GINO:
            self.irr_enc_A = GINO_Encoder_wrapper(in_channels = config_A.in_channels,
                projection_channels=self.projection_channels,                                
                gno_coord_dim = 2)
            self.irr_enc_U = GINO_Encoder_wrapper(in_channels = config_A.in_channels,
                projection_channels=self.projection_channels,                                
                gno_coord_dim = 2)
            self.latent_queries = self.get_latent_grid(128)
        elif self.irr_input and (not self.use_GINO):
            self.irr_enc_A = TinyBlock(in_channels=config_A.in_channels + 1,out_channels=self.projection_channels)
            self.irr_enc_U = TinyBlock(in_channels=config_A.in_channels + 1,out_channels=self.projection_channels)
        
        self.r_enc_A = TinyBlock(in_channels=config_A.in_channels,out_channels=self.projection_channels)
        self.r_enc_U = TinyBlock(in_channels=config_A.in_channels,out_channels=self.projection_channels)
        
        if config_A.use_time:
            self.time_embedding = TimestepEmbedder(1,hidden_size = 768,frequency_embedding_size = 320)
        if config_A.use_cond:
            self.cond_embedding = TimestepEmbedder(self.cond_num,hidden_size = 768,frequency_embedding_size = 256)


        self.model_A = CLIPVisionModel(config_A)
        
        
        self.use_residual = config_U.use_residual
        self.model_U = CLIPVisionModel(config_U)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    @torch.no_grad()
    def encode_A(self,batch,cond = None,timesteps = None,inference_dict = None,Full_msg = False):
        if self.irr_input and (not Full_msg):
            batch = self.encode_irr(batch,"A",inference_dict = inference_dict)
        else:
            batch = self.r_enc_A(batch)
        if cond is not None:
            cond_emb = self.cond_embedding(cond)
            emb = (None,cond_emb)
        else:
            emb = None 
        embed_A  = self.model_A(batch,emb)
        return embed_A

    
    @torch.no_grad()
    def encode_whole_steps(self,batch,silde):
        max_len = 11
        output_list = []
        output_A , output_U = None ,None
        for i,data in enumerate(batch):
            silde_len = silde[i]
            data = data[silde_len:silde_len+1]
            timesteps = [abs(i-silde_len) for i in range(max_len) if i!= silde_len]
            timesteps = torch.tensor(timesteps,device=data.device,dtype=torch.float32)
            if silde_len!=0:
                A_input = data.expand(silde_len,*data.shape[-2:]).unsqueeze(1)
                output_A = self.encode_A(A_input,timesteps[:silde_len]).last_hidden_state 
            if silde_len!=max_len-1:    
                U_input = data.expand(max_len-silde_len-1,*data.shape[-2:]).unsqueeze(1)
                output_U = self.encode_U(U_input,timesteps[silde_len:]).last_hidden_state
            if output_A is not None and output_U is not None:
                output = torch.cat([output_A,output_U])
            else:
                output = output_A if output_A is not None else output_U
            output_list.append(output)
            output = torch.cat(output_list)
        return output
    
    
    @torch.no_grad()
    def encode_all_ticks(self,batch,ticks = None):
        num_tstep = batch.shape[1]
        output_list = []
        for i,data in enumerate(batch):
            tick = ticks[i]
            data = data[tick:tick+1].repeat(num_tstep,1,1).unsqueeze(1)
            timesteps = [i-tick+num_tstep for i in range(num_tstep)]
            timesteps = torch.tensor(timesteps,device=data.device,dtype=torch.float32)
            emb = self.time_embedding(timesteps)
            output = self.model_A(data,t_emb = (emb,)).last_hidden_state
            output_list.append(output)
        output = torch.cat(output_list)
        return output
    
    @torch.no_grad()
    def encode_U(self,batch,cond = None,timesteps = None,inference_dict = None,Full_msg = False):
        if self.irr_input and (not Full_msg):
            batch = self.encode_irr(batch,"U",inference_dict = inference_dict)
        else:
            batch = self.r_enc_U(batch)
        if cond is not None:
            cond_emb = self.cond_embedding(cond)
            emb = (None,cond_emb)
        else:
            emb = None 
        embed_U  = self.model_U(batch,emb)
        return embed_U
    
    def get_latent_grid(self, N):
        xx = torch.linspace(0, 1, N)
        yy = torch.linspace(0, 1, N)

        xx, yy = torch.meshgrid(xx, yy, indexing='ij')
        latent_queries = torch.stack([xx, yy], dim=-1)
        
        return latent_queries.unsqueeze(0)    
    
    def encode_irr(self,data,tag,inference_dict = None):
        """Encode sparse/irregular observations with GINO or masked convolution blocks."""
        module = self.irr_enc_A if tag=="A" else self.irr_enc_U 
        b,c,h,w = data.shape
        if inference_dict is None:
            mask, batch_dict = batch_random_mask_vec(data,b,self.obs_num,h,device=data.device)
        else:
            batch_dict = inference_dict
            mask = None
            
        if self.use_GINO:
            if self.latent_queries.device is not data.device:
                self.latent_queries = self.latent_queries.to(data.device) 
            pos = batch_dict["position"]
            v = batch_dict["values"]
            data = module(v,pos,self.latent_queries)
        else:
            mask_data = torch.ones_like(data)
            mask_data[mask] = data[mask]
            v = torch.cat([mask_data,mask],dim = 1)
            data = module(v)
        return data

    def encode_r(self,data):
        data = self.r_enc(data)
        return data
    
    def forward(self,batch):
        if self.irr_input:
            if torch.rand(1) > 0.5:
                tag = "a2u"
                batch["A"] = self.encode_irr(batch["A"],"A")
                batch["U"] = self.r_enc_U(batch["U"])
            else:
                tag = "u2a"
                batch["A"] = self.r_enc_A(batch["A"])
                batch["U"] = self.encode_irr(batch["U"],"U")
        else:
            batch["U"] = self.r_enc_U(batch["U"])
            batch["A"] = self.r_enc_A(batch["A"])
        if batch.get('timesteps') is not None:
            timesteps = batch['timesteps']
            num_tstep = batch['U'].shape[1]
            input_data_U = batch['U'].flatten(0,1)
            if len(input_data_U) < 4:
                input_data_U = input_data_U.unsqueeze(1)
            else:
                input_data_U = input_data_U.permute(0,3,1,2)
                
            if batch.get('A') is not None:
                if len(input_data_U) < 5:
                    input_data_A = batch['A'].repeat(1,num_tstep,1,1).flatten(0,1).unsqueeze(1)
                else:
                    input_data_A = batch['A'].repeat(1,num_tstep,1,1,1).flatten(0,1).permute(0,3,1,2)
            else:
                ticks = batch['ticks']
                input_data_A = torch.stack([batch['U'][i,ticks[i]:ticks[i]+1,...] for i in range(batch['U'].shape[0])])
                input_data_A = input_data_A.repeat(1,num_tstep,1,1).flatten(0,1).unsqueeze(1)
            
            emb = self.time_embedding(timesteps)
            if batch.get('cond') is not None:
                cond_emb = self.cond_embedding(batch["cond"])
                emb =(emb,cond_emb)
            else:
                emb =(emb,)
            if self.use_residual:
                input_data_U = torch.stack([input_data_U,input_data_U-input_data_A],dim = 1).flatten(1,2)
            embed_U = self.model_U(input_data_U).pooler_output
            embed_A  = self.model_A(input_data_A,t_emb = emb).pooler_output
        else:    
            if batch.get('cond') is not None:
                cond_emb = self.cond_embedding(batch["cond"])
                emb =(None,cond_emb)
                if tag == "u2a":
                    embed_U = self.model_U(batch["U"],emb).pooler_output
                    embed_A  = self.model_A(batch["A"]).pooler_output
                else:
                    embed_U = self.model_U(batch["U"]).pooler_output
                    embed_A  = self.model_A(batch["A"],emb).pooler_output
                    
            else:
                embed_U = self.model_U(batch["U"]).pooler_output
                embed_A  = self.model_A(batch["A"]).pooler_output
        logit_scale = self.logit_scale.exp()
        embed_A = embed_A / embed_A.norm(dim=1, keepdim=True)
        embed_U = embed_U / embed_U.norm(dim=1, keepdim=True)
        logits_U = logit_scale*embed_U@embed_A.T 
        logits_A = logits_U.t()
        return logits_U,logits_A
    
    
def batch_random_mask_vec(data, batch_size, k, grid_size, seed=0, device=None):
 

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


