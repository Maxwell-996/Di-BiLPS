import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from modules.integral_transform import IntegralTransform, MLPLinear
from modules.neighbor_search import NeighborSearch
from modules.embedding import FourierEmb


class GINO_Encoder(nn.Module):
    """GINO: Geometry-informed Neural Operator

        Parameters
        ----------
        in_channels : int
            feature dimension of input points
        out_channels : int
            feature dimension of output points
        projection_channels : int, optional
            number of channels in FNO pointwise projection
        gno_coord_dim : int, optional
            geometric dimension of input/output queries, by default 3
        gno_coord_embed_dim : int, optional
            dimension of positional embedding for gno coordinates, by default None
        gno_embed_max_positions : int, optional
            max positions for use in gno positional embedding, by default None
        gno_radius : float, optional
            radius in input/output space for GNO neighbor search, by default 0.033
        gno_mlp_hidden_layers : list, optional
            widths of hidden layers in input GNO, by default [80, 80, 80]
        gno_mlp_non_linearity : nn.Module, optional
            nonlinearity to use in gno MLP, by default F.gelu
        gno_transform_type : str, optional
            transform type parameter for output GNO, by default 'linear'
            see neuralop.layers.IntegralTransform
        gno_use_torch_scatter : bool, optional
            whether to use torch_scatter's neighborhood reduction function
            or the native PyTorch implementation in IntegralTransform layers.
            If False, uses the fallback PyTorch version.
        """
    def __init__(
            self,
            in_channels,
            projection_channels=256,
            gno_coord_dim=3,
            gno_coord_embed_dim=None,
            gno_radius=0.05,
            gno_mlp_hidden_layers=[64, 64, 64],
            gno_mlp_non_linearity=F.gelu, 
            gno_transform_type='linear',
            gno_use_torch_scatter=True,
            use_open3d=False,
        ):
        
        super().__init__()
        self.in_channels = in_channels
        self.projection_channels = projection_channels
        self.gno_coord_dim = gno_coord_dim
        
        self.nb_search_out = NeighborSearch(use_open3d=use_open3d)
        self.gno_radius = gno_radius

        self.x_projection = MLPLinear(layers=[in_channels, projection_channels])

        if gno_coord_embed_dim is not None:
            self.pos_embed = FourierEmb(hidden_dim=gno_coord_embed_dim, in_dim=gno_coord_dim)
            self.gno_coord_dim = gno_coord_embed_dim 
        else:
            self.pos_embed = None
        
        ### input GNO
        # input to the first GNO MLP: x pos encoding, y (integrand) pos encoding
        in_kernel_in_dim = self.gno_coord_dim * 2
        # add f_y features if input GNO uses a nonlinear kernel
        if gno_transform_type == "nonlinear" or gno_transform_type == "nonlinear_kernelonly":
            in_kernel_in_dim += self.projection_channels
        mlp_layers =  gno_mlp_hidden_layers.copy()
        mlp_layers.insert(0, in_kernel_in_dim)
        mlp_layers.append(projection_channels) 
        self.gno_in = IntegralTransform(
                    mlp_layers=mlp_layers,
                    mlp_non_linearity=gno_mlp_non_linearity,
                    transform_type=gno_transform_type,
                    use_torch_scatter=gno_use_torch_scatter
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, input_geom, latent_queries):
        """forward pass of GNO --> latent embedding w/FNO --> GNO out

        Parameters
        ----------
        x : torch.Tensor
            input function a defined on the input domain `input_geom`
            shape (1, n_in, in_channels) 
        input_geom : torch.Tensor
            input domain coordinate mesh
            shape (1, n_in, gno_coord_dim)
        latent_queries : torch.Tensor
            latent geometry on which to compute FNO latent embeddings
            a grid on [0,1] x [0,1] x ....
            shape (1, n_gridpts_1, .... n_gridpts_n, gno_coord_dim)
        """
        batch_size = x.shape[0]
        input_geom = input_geom.squeeze(0) 
        latent_queries = latent_queries.squeeze(0)
    
        spatial_nbrs = self.nb_search_out(input_geom, 
                                          latent_queries.view((-1, latent_queries.shape[-1])), 
                                          radius=self.gno_radius)
        x = self.x_projection(x) # b, n_in, in_channels -> b, n_in, projection_channels
        x = x.squeeze(0) # n_in, projection_channels

        if self.pos_embed is not None:
            input_geom = self.pos_embed(input_geom) # n_in, coord_dim -> n_in, gno_coord_dim_embed
            latent_queries = self.pos_embed(latent_queries) # n_gridpts_1, .... n_gridpts_n, coord_dim -> n_gridpts_1, .... n_gridpts_n, gno_coord_dim_embed

        in_p = self.gno_in(y=input_geom, # n_in, gno_coord_dim
                           x=latent_queries.view((-1, latent_queries.shape[-1])), # (n_gridpts_1, .... n_gridpts_n), gno_coord_dim
                           f_y=x, # b, n_in, projection_channels
                           neighbors=spatial_nbrs)
        
        grid_shape = latent_queries.shape[:-1] 
        in_p = in_p.view((batch_size, *grid_shape, self.projection_channels))
        in_p = in_p.permute(0,3,1,2)
        return in_p # shape [1, n_gridpts_1, ..., n_gridpts_n, f_dim]


class GINO_Decoder(nn.Module):
    """GINO: Geometry-informed Neural Operator

        Parameters
        ----------
        out_channels : int
            feature dimension of output points
        projection_channels : int, optional
            number of channels in FNO pointwise projection
        gno_coord_dim : int, optional
            geometric dimension of input/output queries, by default 3
        gno_coord_embed_dim : int, optional
            dimension of positional embedding for gno coordinates, by default None
        gno_embed_max_positions : int, optional
            max positions for use in gno positional embedding, by default None
        gno_radius : float, optional
            radius in input/output space for GNO neighbor search, by default 0.033
        gno_mlp_hidden_layers : list, optional
            widths of hidden layers in output GNO, by default [512, 256]
        gno_mlp_non_linearity : nn.Module, optional
            nonlinearity to use in gno MLP, by default F.gelu
        gno_transform_type : str, optional
            transform type parameter for output GNO, by default 'linear'
            see neuralop.layers.IntegralTransform
        gno_use_torch_scatter : bool, optional
            whether to use torch_scatter's neighborhood reduction function
            or the native PyTorch implementation in IntegralTransform layers.
            If False, uses the fallback PyTorch version.
        out_gno_tanh : bool, optional
            whether to use tanh to stabilize outputs of the output GNO, by default False
        """
    def __init__(
            self,
            out_channels = 1,
            projection_channels=256,
            gno_coord_dim=3,
            gno_coord_embed_dim=None,
            gno_radius=0.01,
            gno_mlp_hidden_layers=[128, 64],
            gno_mlp_non_linearity=F.gelu, 
            gno_transform_type='linear',
            gno_use_torch_scatter=True,
            use_open3d=False,
            tanh_out = False,
            use_2_head = False,
        ):
        
        super().__init__()
        self.out_channels = out_channels
        self.gno_coord_dim = gno_coord_dim
        self.use_2_head = use_2_head
        self.nb_search_out = NeighborSearch(use_open3d=use_open3d)
        self.gno_radius = gno_radius
        self.tanh_out = tanh_out

        if gno_coord_embed_dim is not None:
            self.pos_embed = FourierEmb(hidden_dim=gno_coord_embed_dim, in_dim=gno_coord_dim)
            self.gno_coord_dim = gno_coord_embed_dim 
        else:
            self.pos_embed = None

        ### output GNO
        out_kernel_in_dim = 2 * self.gno_coord_dim
        out_kernel_in_dim += projection_channels if gno_transform_type != 'linear' else 0
        mlp_layers = gno_mlp_hidden_layers.copy()
        mlp_layers.insert(0, out_kernel_in_dim)
        mlp_layers.append(projection_channels)
        self.gno_out = IntegralTransform(
                    mlp_layers=mlp_layers,
                    mlp_non_linearity=gno_mlp_non_linearity,
                    transform_type=gno_transform_type,
                    use_torch_scatter=gno_use_torch_scatter
        )
        self.projection = MLPLinear(layers=[projection_channels, out_channels]) 
        if self.use_2_head:
            self.projectionA = MLPLinear(layers=[projection_channels, 1]) 
            self.projectionU = MLPLinear(layers=[projection_channels, 1]) 
        else:
            self.projection = MLPLinear(layers=[projection_channels, out_channels]) 

    # out_p : (n_out, gno_coord_dim)

    def integrate_latent(self, in_p, out_p, latent_embed):
        # in_p is in shape (n_gridpts_1, .... n_gridpts_n, gno_coord_dim)
        # out_p is in shape (n_out, gno_coord_dim)

        in_to_out_nb = self.nb_search_out(
            in_p.view(-1, in_p.shape[-1]), 
            out_p,
            self.gno_radius,
            )# for each output point, find the neighbors in the latent grid 
    
        #Embed input points
        n_in = in_p.view(-1, in_p.shape[-1]).shape[0]
        in_p_embed = in_p.reshape((n_in, -1)) # flatten to ((n_gridpts_1, .... n_gridpts_n), gno_coord_dim)
        if self.pos_embed is not None:
            in_p_embed = self.pos_embed(in_p_embed)
        
        #Embed output points
        out_p_embed = out_p
        if self.pos_embed is not None:
            out_p_embed = self.pos_embed(out_p_embed)
        
        latent_embed = rearrange(latent_embed, 'b n1 n2 c -> b (n1 n2) c')
        # rehape to batch, (n_1 * n_2 * ... * n_k), hidden_channels

        #(n_out, fno_hidden_channels)
        out = self.gno_out(y=in_p_embed, 
                    neighbors=in_to_out_nb,
                    x=out_p_embed,
                    f_y=latent_embed,) # apply kernel integration to latent embedding, sum on output points
        
        
        if self.tanh_out:
            out = torch.tanh(out)

        return out
    
    def forward(self, latent_embed, latent_queries, output_queries):
        """forward pass of GNO --> latent embedding w/FNO --> GNO out

        Parameters
        ----------
        latent_embed : torch.Tensor
            latent_embedding to be decoded
            shape (batch, n_gridpts_1, .... n_gridpts_n, hidden_channels) 
        latent_queries : torch.Tensor
            latent geometry on which to compute FNO latent embeddings
            a grid on [0,1] x [0,1] x ....
            shape (1, n_gridpts_1, .... n_gridpts_n, gno_coord_dim)
        output_queries : torch.Tensor
            points at which to query the final GNO layer to get output
            shape (batch, n_out, gno_coord_dim)
        """
        latent_queries = latent_queries.squeeze(0)
        output_queries = output_queries.squeeze(0)

        out = self.integrate_latent(latent_queries, output_queries, latent_embed) 
        if self.use_2_head:
            out_A = self.projectionA(out)
            out_U = self.projectionU(out)
            out = torch.cat([out_U,out_A],dim = -1)
        else:
            out = self.projection(out)
        return out # shape (batch, n_latent_queries, f_dim)
    

class GINO_Encoder_wrapper(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.irr_Encoder = GINO_Encoder(**kwargs)
        
        
    def forward(self, x, input_geom, latent_queries, pad_mask=None):
        b = x.shape[0]
        
        if b > 1:
            latent = []
            for i in range(b):
                x_batch = x[i].unsqueeze(0) # shape [1, nt, n, in_channels]
                input_geom_batch = input_geom[i].unsqueeze(0) # shape [1, nt, n, gno_coord_dim]
                latent_batch = self.irr_Encoder(x_batch, input_geom_batch, latent_queries) # 1 n_gridpts_1, .... n_gridpts_n, hidden_channels
                latent.append(latent_batch)
            latent = torch.cat(latent, dim=0) # shape [batch, n_gridpts_1, .... n_gridpts_n, hidden_channels]

        else:
            latent = self.irr_Encoder(x, input_geom, latent_queries)
            
        return latent    
    
class GINO_Decoder_wrapper(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.irr_Decoder = GINO_Decoder(**kwargs)
        
        
    def forward(self, latent_embed, latent_queries, output_queries, pad_mask=None):
        
        
        if len(latent_embed.shape) == 4:
            latent_embed = rearrange(latent_embed, 'b c n1 n2 -> b n1 n2 c')
        else:
            latent_embed = rearrange(latent_embed, 'b c n1 n2 n3 -> b n1 n2 n3 c')
        
        b = latent_embed.shape[0]
        
        if b > 1:
            out = []
            for i in range(b):
                latent_embed_batch = latent_embed[i].unsqueeze(0) 
                out_batch = self.irr_Decoder(latent_embed_batch, latent_queries, output_queries) # 1 n_gridpts_1, .... n_gridpts_n, hidden_channels
                out.append(out_batch)
            out = torch.cat(out, dim=0) # shape [batch, n_gridpts_1, .... n_gridpts_n, hidden_channels]

        else:
            out = self.irr_Decoder(latent_embed, latent_queries, output_queries)
        return out
