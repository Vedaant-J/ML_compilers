import time
import json
import torch
import torch.nn.functional as F
from myconv import ConvModel
import argparse
from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()

    torch.manual_seed(0)

    N, C, H, W = 1, 3, args.h, args.w
    out_channels = 8
    kernel_size = args.k
    
    key = f"H_W_{H}_K_{kernel_size}"
    print(f"--- Profiling PyTorch for {key} ---")

    x = torch.randn(N, C, H, W).cuda()
    model = ConvModel(H, W, in_channels=C, out_channels=out_channels, kernel_size=kernel_size).cuda().eval()
    
    print("Warming up...")
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize()
    
    print("Profiling...")
    wall_time_start = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        y_pytorch = model(x)
        torch.cuda.synchronize()
    
    print(f"Total wall time for inference: {(time.time() - wall_time_start) * 1000:.4f} ms")
    print("Averages:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    
    trace_filename = f"pytorch_trace_{key}.json"
    prof.export_chrome_trace(trace_filename)
