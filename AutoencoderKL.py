from diffusers import AutoencoderKL
from GINO import GINO_Decoder_wrapper
import torch
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.autoencoders.vae import DecoderOutput
from typing import  Dict, Optional, Tuple, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config

class GINOAutoencoderKL(AutoencoderKL):
    """AutoencoderKL variant with a GINO decoder for irregular output fields."""

    @register_to_config
    def __init__(self,   
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: float = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
        var_num: int = 1,
        multi_pre: bool = False,
        use_2_head = False):
        
        super().__init__(         
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            latents_mean=latents_mean,
            latents_std=latents_std,
            force_upcast=force_upcast,
            use_quant_conv=use_quant_conv,
            use_post_quant_conv=use_post_quant_conv,
            mid_block_add_attention=mid_block_add_attention,
            multi_pre = multi_pre,
            )
        self.irr_dec = GINO_Decoder_wrapper(
            out_channels = 2,
            projection_channels=self.out_channels,
            gno_coord_dim = 2,
            use_2_head = use_2_head
        )
        self.latent_queries = self.get_latent_grid(128)
        
    def get_latent_grid(self, N):
        """Create normalized 2D query coordinates for the GINO decoder."""
        xx = torch.linspace(0, 1, N)
        yy = torch.linspace(0, 1, N)

        xx, yy = torch.meshgrid(xx, yy, indexing='ij')
        latent_queries = torch.stack([xx, yy], dim=-1)
        
        return latent_queries.unsqueeze(0)  
    
    @apply_forward_hook
    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        if z.shape[1] != self.latent_channels:
            b,c,h,w = z.shape
            z = z.view(-1,self.latent_channels,h,w)    
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample
        if self.latent_queries.device is not decoded.device:
                self.latent_queries = self.latent_queries.to(decoded.device) 
        b,c,h,w = decoded.shape
        # Query the learned latent grid and reshape it back to a dense 2-channel field.
        query_pos = self.latent_queries.reshape(1,h*h,2)  
        decoded = self.irr_dec(latent_embed = decoded,latent_queries = self.latent_queries,output_queries = query_pos)
        decoded =  decoded.reshape(b,h,w,2).permute(0,3,1,2)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
        