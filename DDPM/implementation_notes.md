# **Implementation notes for Denoising Diffusion Probabilistic Models**

## Resnet Blocks
```
Our neural network architecture follows the backbone of PixelCNN++ [52], which is a U-Net [48]
based on a Wide ResNet [72]. We replaced weight normalization [49] with group normalization [66]
to make the implementation simpler. Our 32 × 32 models use four feature map resolutions (32 × 32
to 4 × 4), and our 256 × 256 models use six. All models have two convolutional residual blocks
per resolution level and self-attention blocks at the 16 × 16 resolution between the convolutional
blocks [6]. Diffusion time t is specified by adding the Transformer sinusoidal position embedding [60]
into each residual block. 
```

There are a bunch of details here that matter. First of all, U-Net consists of residual blocks of 2 3x3 convolution (with padding=1 to simplify residual connections in the U-net, even though in the original U-net there is no padding). Even though you could implement something like 2*(conv-norm-FiLM-act), in the original implementation they do something a bit different. First, they assume that you first have a convolution layer that goes from original image channels to network channels, and then the architecture looks like this: (norm-act-conv)-(norm-FiLM-act-dropout-conv). So they only really apply FiLM once per resnet block, and they add dropout only in the second conv block. Additionally, the output of this block has a residual connection, such that if the output of the above combined architecture is h, then the output of the layer is `h + skip_connection(x)`, where x is the original input, `skip_connection` is either an identity layer if input channels is the same as output channels, or a convolutional layer with 1x1 kernel if input channels is different than output channels.


Regarding choice of activation, they don't specify which activation in the DDPM paper. Even though the wide resnet paper uses ReLU, in the original codebase they use Swish, which is $\text{swish}_{\beta}(x) = x \cdot \text{sigmoid}(\beta x)$ but with $\beta=1$ which is the same as the SiLU activation, which is the activation used in the Improved DDPM paper). 

It also seems like the number of groups in the original DDPM paper and in the improved DDPM paper is 32. In the annotated DDPM paper, they replace the regular convolutional layer with a weight-standardized convolutional layer, which roughly subtracts the mean and divides by the std of the weights, but I will implement this later. 


## Timestep Embeddings

In the improved DDPM codebase, timesteps are converted first into sinusoidal embeddings using the same formula as transformers, and are then passed through a shared MLP: (timesteps-sinusoidal_emb-linear-silu-linear). Interestingly, say the model's channel dimension is d, the first linear layer will have dimensions `(in=d, out=4*d)`, and the next linear layer will have dimensions `(in=4*d, out=4*d)`. Then, within each resnet block, there will be an MLP with order (silu, linear(in=4*d, out=d)) which resizes to the right output dimension needed so that these timestep embeddings can be added to the resnet block.

For me, it is a bit tough to parse what the positional embeddings are supposed to be, given that $i$ and $d_{\text{model}}$ are not well-defined in the original transformer paper. It is easier to think of $i$ as the pair index, where the final embedding will have $2i$ pairs of elements, each element consisting of $(\sin, \cos)$. So say the desired embedding dimension is $4$, then there will be two $i$'s: $(0, 1)$. Then, we can compute all denominators all at once given these $i$'s.

## On TimestepBlock and TimestepEmbedSequential

In the original implementation, the authors define these two classes. The point of `TimestepBlock` is to simply define a module which will take as input not only `x` but also `time_emb`. This will be used for checking it any given module in a `Sequential` block is a `TimestepBlock` or not, and the job of `TimestepEmbedSequential` is to effectively do an if statement for each layer in the `Sequential` block, and if the layer is a `TimestepBlock`, then run it with the `x` and `time_emb` inputs, otherwise just run with the `x` inputs. 

## Downsampling

The original U-net code uses 2x2 maxpool operations with a stride of 2 to downsample a residual block, and also doubles the number of channels. However, in the improved DDPM implementation, there are two options. The default option (if `use_conv=True`) is to use a learned convolutional layer to downsample by using a stride of 2 and 3x3 kernel with padding=1. The other option is to simply use an `AvgPool2d` operation with kernel size 2. Interestingly, this downsample operation keeps the number of channels constant, and its only job is to downsample the height and width dimensions without also changing the channel dimension. To change channel dimension, the original codebase uses another ResNet block.

## Upsampling 

To upsample, the original codebase uses `nn.functional.interpolation` with `scale_factor=2` and `mode=nearest` to upsample the height and width, rather than using something like a `ConvTranspose2d`. Optionally, the upsampled features also go through a convolutional layer with the same input and output channels, 3x3 kernel, and padding of 1.

## Attention Block

DDPM introduces an attention block in three places: in the downsampling block at the 16x16 resolution ResNet block, in the middle bottleneck, and in the upsampling block at the 16x16 resolution. To implement the attention blocks, we can split up the QKV Attention logic (from the Attention is all you need paper), and the logic surrounding the QKV attention block, e.g. flattening H,W dimensions, creating the QKV tensors, adding normalization and residual connection, etc. 

## QKV Attention
One important design decision is how we convert the output of a convolutional layer (B, C, H, W) into the shapes needed for the attention operation, which expects (B, L, D), where L is the sequence length and D is the dimensionality of each datapoint in the sequence. In this case, we want the attention operator to have the job of determining what other pixels in the "image" each pixel should pay attention to, so we can think of the HxW dimensions combined as the "sequence length" and the channel dimension C as the feature dimension D.

So in order to convert (B, C, H, W), we first flatten the H, W dimensions into L, to get a tensor (B, C, L). The goal is to eventually obtain tensors Q, K, and V, each of which has dimensionality (B, L, C) from the input with dimensionality (B, C, L). To do this, we have two options. First, we could reshape the (B, C, L) input into (B, L, C) and create a linear layer with input dim C and output dim 3C, since linear layers operate on the last dimension and. Another option is to use a 1-dim convolution which operates on the second dimension, C, and use `in_channels=C` and `out_channels=3C`, and then reshape this (B, 3C, L) tensor into (B, L, 3C) either before or after splitting these 3C into the respective Q, K, V tensors. The original ddpm implementation uses a 1-dim convolution for this operation, so that's what I will use too. An important note is that the Q, K, V tensors expect this order (B, L, C), which matters when deciding which dimension the softmax should operate over. If we think of L as the sequence of pixels, The `QK^T` matrix is a (B, L, L) tensor where the L, L dims form a square matrix, where each row i will correspond to the probability that a given pixel should pay attention to each of the other pixels, so the softmax operation needs to happen for each row, which corresponds to the `-1` dimension, i.e. the last L in (B, L, L). Otherwise, if you operate in the (B, C, L) order directly, the attention operator will give you the probability that each dimension of the channel should attend at other channel dimensions for all elements of the sequence, which doesn't make sense. 

## Attention Block

Once we have the QKV attention implemented, we need the tooling around it. This involves: normalizing the input [PreNorm](https://tnq177.github.io/assets/docs/transformers_without_tears.pdf), getting the QKV tensor from the input, adding multi-head attention logic, running the QKV Attention operation, concatenating the multiple heads, projecting the output of attention with a 1D convolutional layer, and adding the residual connection. There are two things that I found interesting:
  1. The first thing is that normalization needs to happen before we project the input into the QKV tensor. This is because otherwise, the normalization would make Q, K, and V correlated.
  2. Second, to implement Multi-Head Attention, we can simply divide the channel dimension by the number of heads, and do batch folding, in which we concatenate the heads along the batch dimension, run the same attention mechanism as if you had done single-head attention, and then after the attention operation, reshape into the right dimensions.

## UNet