from abc import abstractmethod
import time
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
    log_denoms = 2*torch.arange(0, num_emb_pairs, device=timesteps.device)/emb_dim*math.log(torch.tensor(10000))  # Should be (emb_dim//2,)
    angles = timesteps @ (1/torch.exp(log_denoms.reshape(1, -1)))  # Should be (n, emb_dim//2)
    sines = torch.sin(angles)
    cosines = torch.cos(angles)
    embeddings = torch.cat((sines, cosines), dim=1)  # This just concatenates, but technically they should be interleaved

    return embeddings

## NOTE: Implemented this third
## We are going to define a TimestepBlock class so that we can run isinstance() with classes that take in time_emb and not just x
## Then, we'll implement TimestepEmbedSequential which inherits from Sequential but adds an if statement that chooses whther to pass only x or x and time_emb depending on each layer's inputs
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, time_emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, time_emb):
        for module in self:
            if isinstance(module, TimestepBlock):
                x = module(x, time_emb)
            else:
                x = module(x)
        return x

## NOTE: Implemented this fourth
class Downsample(nn.Module):
    def __init__(self, num_channels, use_conv):
        if use_conv:
            self.downsample_layer = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample_layer = nn.AvgPool2d(kernel_size=2) 

    def forward(self, x):
        return self.downsample_layer(x)

class Upsample(nn.Module):
    def __init__(self, num_channels, use_conv):
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


## NOTE: Implemented this fifth
class QKVAttention(nn.Module):
    def forward(self, qkv):
        '''
        :param qkv: a (B, 3C, HxW) tensor of Qs, Ks, and Vs
        :return: a (B, C, HxW) tensor after self-attention 
        '''
        channel_dim = qkv.shape[1]//3
        Q, K, V = torch.split(qkv, [channel_dim]*3, dim=1)  # Each should have dimensions (B, C, HxW)
        Q, K, V = Q.permute(0, 2, 1), K.permute(0, 2, 1), V.permute(0, 2, 1)
        
        attention = nn.functional.softmax((1/math.sqrt(channel_dim))*(Q@torch.transpose(K, 1, 2)), dim=-1)@V
        output = attention.permute(0, 2, 1)

        return output

class AttentionBlock(nn.Module):
    def __init__(self, num_channels, heads=1, groups=32):
        self.num_heads = heads
        self.qkv = nn.Conv1d(in_channels=num_channels, out_channels=3*num_channels, kernel_size=1)
        self.proj_out = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)
        self.attention = QKVAttention()
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=num_channels)

    def forward(self, x):
        # Flatten H, W dimensions
        b, c, *spatial_dims = x.shape  # x is (B, C, H, W)
        x = x.reshape(b, c, -1)  # (B, C, L), L=H*W
        # Normalize and Get QKV weights from input x:
        qkv = self.qkv(self.norm(x))  # (B, 3C, L)
        # Batch folding, i.e., split channels into different heads
        qkv = qkv.reshape(b*self.num_heads, -1, qkv.shape[2])  # (B*num_heads, 3C//num_heads, L)
        # Run normalization and attention (in PreNorm order from https://arxiv.org/pdf/1910.05895)
        attention = self.attention(qkv)  # (B*num_heads, C//num_heads, L)
        # Concatenate multiple heads
        attention = attention.reshape(b, -1, attention.shape[-1])  # (B, C, L)
        # Project the attention output
        output = self.proj_out(attention)  # (B, C, L)
        # Run residual connection and unflatten
        output = (output + x).reshape(b, c, *spatial_dims)  # (B, C, H, W)

        return output


## NOTE: Could make this a more general unet ddpm by defining the ResnetBlock wrt a conv dim, and having a helper function which defines the conv1d, conv2d, or conv3d based on the dim
## NOTE: Implemented this first
class ResnetBlock(TimestepBlock):
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
        
## NOTE: Implemented this sixth
class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        model_channels,
        channel_multipliers,
        num_res_blocks,
        groups,
        dropout,
        use_scale_shift,
        attention_resolutions,
        use_conv,
        num_heads
        ):
        '''
        :param in_channels: An int for the number of channels in the input data (3 for RGB images)
        :param out_channels: An int for the number of output channels (3 if learning only mean, 6 if learning std)
        :param model_channels: Initial channel dimension of the model
        :param channel_multipliers: A list of multipliers, one per level, that the initial num_channels will be multiplied to
        :param groups: Number of groups for group normalization. The DDPM paper sets this to 32
        :param num_res_blocks: int of th enumber of resnet blocks to be applied per level
        :param dropout: float [0, 1] of dropout probability
        :param use_scale_shift: Bool of whether to use FiLM (scale + shift) for time embedding or just add
        :param attention_resolutions: a list of resolutions (in spatial dims) at which an attention block should be applied 
        :param use_conv: boolean of whther to use a convolutional layer for downsampling or upsampling, or use non-learned operations only (AvgPool for downscale and interpolate for upsample)
        :param num_heads: Number of heads for Multi-Head Attention
        '''
        super().__init__()


        self.model_channels = model_channels

        # Shared time projector
        time_emb_dim = 4 * model_channels
        self.shared_time_proj = nn.Sequential(
            nn.Linear(in_features=model_channels, out_features=time_emb_dim),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dim, out_features=time_emb_dim)
        )

        # Downsampling block
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    # Initial input projector
                    nn.Conv2d(in_channels=in_channels, out_channels=model_channels, kernel_size=3, padding=1)
                )
            ]
        )
        input_blocks_channels = [model_channels]

        level_channels = model_channels
        ds = 1  # Downsampling scale

        for level, mult in enumerate(channel_multipliers):
            for j in range(num_res_blocks):
                layers = []
                # Create Resnet Block
                layers.append(
                    ResnetBlock(
                        in_channels=level_channels, 
                        out_channels=mult*model_channels, 
                        groups=groups,
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                        use_scale_shift=use_scale_shift)
                    )
                level_channels = mult*model_channels

                # Create Attention block if at the right resolution
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(level_channels, num_heads, groups)
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_blocks_channels.append(level_channels)
            if level != len(channel_multipliers) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(level_channels, use_conv)
                    )
                )
                input_blocks_channels.append(level_channels)
                ds *= 2

        # Middle Block:
        self.middle_block = TimestepEmbedSequential(
            ResnetBlock(
                in_channels=level_channels,
                out_channels=level_channels,
                groups=groups,
                time_emb_dim=time_emb_dim,
                dropout=dropout,
                use_scale_shift=use_scale_shift
            ),
            AttentionBlock(
                num_channels=level_channels,
                heads=num_heads,
                groups=groups
            ),
            ResnetBlock(
                in_channels=level_channels,
                out_channels=level_channels,
                groups=groups,
                time_emb_dim=time_emb_dim,
                dropout=dropout,
                use_scale_shift=use_scale_shift
            )
        )

        # Need an upsampling block
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_multipliers))[::-1]:
            for j in range(num_res_blocks + 1):
                layers = []
                # Create Resnet Block
                layers.append(
                    ResnetBlock(
                        in_channels=level_channels+input_blocks_channels.pop(), 
                        out_channels=mult*model_channels, 
                        groups=groups,
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                        use_scale_shift=use_scale_shift)
                    )
                level_channels = mult*model_channels

                # Create Attention block if at the right resolution
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(level_channels, num_heads, groups)
                    )
                if level != 0 and j == num_res_blocks:
                    layers.append(
                        Upsample(num_channels=level_channels, use_conv=use_conv)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(groups, num_channels=level_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=level_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        '''
        :param x: a (B, 3, H, W) tensor
        :param t: a (B, 1) tensor of timesteps
        '''
        # Embed the timesteps using sinusoidal positional encoding + shared projector, to get (B, 4*D), where D comes from sinusoidal encoding, 4 comes from the shared projection
        encoded_t = time_embedding(t, self.model_channels)
        time_emb = self.shared_time_proj(encoded_t)

        # pass through ModuleList
        hs = []
        for module in self.input_blocks:
            x = module(x, time_emb)
            hs.append(x)

        x = self.middle_block(x, time_emb)

        for i, module in enumerate(self.output_blocks):
            skip_connection = hs.pop()
            x = torch.cat([x, skip_connection], dim=1)
            x = module(x, time_emb)

        out = self.out(x)

        return out

        


if __name__ == "__main__":
    timesteps = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    emb_dim = 4

    time_emb = time_embedding(timesteps, emb_dim)
    print(f"Time embeddings")
    print(time_emb)