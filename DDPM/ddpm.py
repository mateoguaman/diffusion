import torch
import torch.nn as nn
import math

## NOTE: Implemented this second
def time_embedding(timesteps, emb_dim):
    '''
    Creates sinusoidal time embeddings similar to Transformer positional embeddings
    :param timesteps: a (n,) tensor of timesteps, one per data point
    :param emb_dim: int, the dimension of the time embeddings
    :return: a (n, emb_dim) tensor of time embeddings
    '''
    timesteps = timesteps.reshape(-1, 1)
    num_emb_pairs = emb_dim // 2
    log_denoms = 2*torch.arange(0, num_emb_pairs)/emb_dim*math.log(torch.tensor(10000))  # Should be (emb_dim//2,)
    angles = timesteps @ (1/torch.exp(log_denoms.reshape(1, -1)))  # Should be (n, emb_dim//2)
    sines = torch.sin(angles)
    cosines = torch.cos(angles)
    embeddings = torch.cat((sines, cosines), dim=1)  # This just concatenates, but technically they should be interleaved

    return embeddings


## NOTE: Could make this a more general unet ddpm by defining the ResnetBlock wrt a conv dim, and having a helper function which defines the conv1d, conv2d, or conv3d based on the dim
## NOTE: Implemented this first
class ResnetBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        groups, 
        time_emb_dim,
        dropout,
        use_scale_shift=False,
        ):
        super().__init__()
        self.use_scale_shift = use_scale_shift

        self.block_1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # TODO: Change into weight-standardized convolution
        )  # NOTE: Requires there to be an initial conv2d layer that projects the initial 3 channels into something larger 

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2*out_channels if self.use_scale_shift else out_channels),
        )  # NOTE: In the FiLM paper, they mention how it's important for the film layer to go right after normalization.
        
        self.block_2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # TODO: Initialize with zero weights. TODO: Change into weight-standardized convolution
        )

        if in_channels == out_channels:
            self.residual_connection = nn.Identity()
        else:
            self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, time_emb):
        time_emb_proj = self.time_mlp(time_emb)  # This should return a (b, c or 2*c)
        time_emb_proj = time_emb_proj[..., None, None]
        if self.use_scale_shift:
            scale, shift = torch.chunk(time_emb_proj, 2, 1)
        else:
            scale = torch.zeros_like(time_emb_proj)
            shift = time_emb_proj
        
        out = self.block_1(x)
        block_2_norm, block_2_rest = self.block_2[0], self.block_2[1:]  # Manually add the FiLM embedding after the GroupNorm, then run the rest of the sequential block
        out = block_2_norm(out)
        out = (scale + 1) * out + shift
        out = block_2_rest(out)
        out = self.residual_connection(x) + out

        return out
        

if __name__ == "__main__":
    timesteps = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    emb_dim = 4

    time_emb = time_embedding(timesteps, emb_dim)
    print(f"Time embeddings")
    print(time_emb)