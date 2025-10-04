import jax
import jax.numpy as jnp
from jax import jit
import torch.nn.functional as F
import numpy as np
import torch
from myconv import ConvModel
import jax.profiler
import torch

# Create a log directory
logdir = "./jax_trace"

def im2col_manual_jax(x, KH, KW, S, P, out_h, out_w):
    ''' 
        Reimplement the same function (im2col_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
    '''
    # x: (N, C, H, W)
    N, C, H, W = x.shape

    # Pad input
    x_pad = jnp.pad(x, ((0,0),(0,0),(P,P),(P,P))) # (N, C, H+2P, W+2P)

    # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
    # Refer to Lecture 3 for implementing this operation.
    starting_rows = jnp.arange(0, H+2*P-KH+1, step=S)
    starting_cols = jnp.arange(0, W+2*P-KW+1, step=S)
    assert out_h == len(starting_rows)
    assert out_w == len(starting_cols)
    grid_rows = jnp.repeat(starting_rows, out_w)
    grid_cols = jnp.tile(starting_cols, out_h)
    row_patches = grid_rows.reshape(-1, 1) + jnp.arange(KH).reshape(1, -1) # (out_h*out_w, KH)
    col_patches = grid_cols.reshape(-1, 1) + jnp.arange(KW).reshape(1, -1) # (out_h*out_w, KW)
    row_patches = row_patches[:, :, jnp.newaxis]  # (out_h*out_w, KH, 1)
    col_patches = col_patches[:, jnp.newaxis, :]  # (out_h*out_w, 1, KW)
    patches = x_pad[:, :, row_patches, col_patches] # (N, C, out_h*out_w, KH, KW)
    patches = jnp.permute_dims(patches, (0, 2, 1, 3, 4)) # (N, out_h*out_w, C, KH, KW)
    patches = patches.reshape(N, out_h*out_w, C*KH*KW)
    # patches = ...
    return patches

def conv2d_manual_jax(x, weight, bias, stride=1, padding=1):
    '''
        Reimplement the same function (conv2d_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
        Hint: Unlike PyTorch, JAX arrays are immutable, so you cannot do indexing like out[i:j, :] = ... inside a JIT. You may use .at[].set() instead.
    '''
    N, C, H, W = x.shape
    C_out, _, KH, KW = weight.shape

    # define your helper variables here
    out_h = (H + 2 * padding - KH) // stride + 1
    out_w = (W + 2 * padding - KW) // stride + 1
    
    # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
    cols = im2col_manual_jax(x, KH, KW, stride, padding, out_h, out_w)

    # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
    weight_flat = weight.reshape(C_out, C*KH*KW)
    weight_flat = jnp.transpose(weight_flat)  # (C*KH*KW, C_out)
    # TO DO: 3) perform tiled matmul after required reshaping is done.
    TILE_SIZE = 16
    out = jnp.zeros((N, out_h*out_w, C_out), dtype=x.dtype)
    for i in range(0, out_h*out_w, TILE_SIZE):
        for j in range(0, C_out, TILE_SIZE):
            for k in range(0, cols.shape[2], TILE_SIZE):
                cols_tile = cols[:, i:i+TILE_SIZE, k:k+TILE_SIZE]  # (N, TILE_SIZE, TILE_SIZE)
                weight_tile = weight_flat[k:k+TILE_SIZE, j:j+TILE_SIZE] # (TILE_SIZE, TILE_SIZE)
                out = out.at[:, i:i+TILE_SIZE, j:j+TILE_SIZE].add(jnp.matmul(cols_tile, weight_tile)) # (N, TILE_SIZE, TILE_SIZE)
    # TO DO: 4) Add bias.
    out = out + bias.reshape(1, 1, -1)

    # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
    out = out.reshape(N, out_h, out_w, C_out)
    out = jnp.transpose(out, (0, 3, 1, 2))

    return out

if __name__ == "__main__":
    # Instantiate PyTorch model
    H, W = 33, 33
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=1)
    model.eval()

    # Example input
    x_torch = torch.randn(1, 3, H, W)

    # Export weights and biases
    params = {
        "weight": model.weight.detach().cpu().numpy(),  # shape (out_channels, in_channels, KH, KW)
        "bias": model.bias.detach().cpu().numpy()       # shape (out_channels,)
    }

    # Convert model input, weights and bias into jax arrays
    x_jax = jnp.array(x_torch.numpy())
    weight_jax = jnp.array(params["weight"])
    bias_jax = jnp.array(params["bias"])

    # enable JIT compilation
    conv2d_manual_jax_jit = jit(conv2d_manual_jax)

    # call your JAX function
    out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax)

    # Test your solution
    conv_ref = F.conv2d(x_torch, model.weight, model.bias, stride=1, padding=1)
    print("JAX --- shape check:", out_jax.shape == conv_ref.shape)
    out_jax = torch.from_numpy(np.array(out_jax))
    print("JAX --- correctness check:", torch.allclose(out_jax, conv_ref, atol=1e-1))
