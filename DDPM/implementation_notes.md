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