import time
import json
import torch
import numpy as np
from myconv_jax import conv2d_manual_jax, ConvModel
import jax
import jax.numpy as jnp
from jax import jit
import argparse
from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    N, C, H, W = 1, 3, args.h, args.w
    out_channels = 8
    kernel_size = args.k

    key = f"H_W_{H}_K_{kernel_size}"
    print(f"--- Profiling JAX for {key} ---")

    torch_model = ConvModel(H, W, in_channels=C, out_channels=out_channels, kernel_size=kernel_size)
    x_torch = torch.randn(N, C, H, W)

    x_jax = jnp.array(x_torch.numpy())
    weight_jax = jnp.array(torch_model.weight.detach().cpu().numpy())
    bias_jax = jnp.array(torch_model.bias.detach().cpu().numpy())

    start_compile_time = time.time()
    conv2d_jit = jit(conv2d_manual_jax)
    _ = conv2d_jit(x_jax, weight_jax, bias_jax).block_until_ready()
    compile_time = time.time() - start_compile_time
    print(f"Compilation complete in: {compile_time:.4f} seconds")

    # for _ in range(5):
    #     _ = conv2d_jit(x_jax, weight_jax, bias_jax).block_until_ready()
    
    print("Profiling...")
    
    wall_start = time.time()
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #     y_jax = conv2d_jit(x_jax, weight_jax, bias_jax)
    #     y_jax.block_until_ready()
    with jax.profiler.trace("jax-traces/", create_perfetto_link=False):
        y_jax = conv2d_jit(x_jax, weight_jax, bias_jax)
        y_jax.block_until_ready()
    wall_time = time.time() - wall_start

    print(f"Total wall time for inference: {wall_time * 1000:.4f} ms")

    # print("Averages:")
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    
    # trace_filename = f"jax_trace_{key}.json"
    # prof.export_chrome_trace(trace_filename)
