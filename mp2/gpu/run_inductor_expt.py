import time
import json
import torch
import torch.nn.functional as F
from myconv import ConvModel
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, required=True, help="Height of the input tensor")
    parser.add_argument("--w", type=int, required=True, help="Width of the input tensor")
    parser.add_argument("--k", type=int, required=True, help="Kernel size")
    args = parser.parse_args()

    torch.manual_seed(0)
    N, C, H, W = 1, 3, args.h, args.w
    out_channels = 8
    kernel_size = args.k
    
    key = f"H/W:{H}_K:{kernel_size}"
    print(f"--- Running Inductor Benchmark for {key} ---")

    x = torch.randn(N, C, H, W).cuda()
    model = ConvModel(H, W, in_channels=C, out_channels=out_channels, kernel_size=kernel_size).cuda().eval()
    
    start_compile_time = time.time()
    
    compiled_model = torch.compile(model, backend="inductor")
    
    
    _ = compiled_model(x) 
    torch.cuda.synchronize()

    compile_time = time.time() - start_compile_time
    print(f"Compile time: {compile_time:.4f} seconds")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    
    for _ in range(5):
        _ = compiled_model(x)

    
    iterations = 10
    start_event.record()
    for _ in range(iterations):
        _ = compiled_model(x)
    end_event.record()

    torch.cuda.synchronize()

    total_gpu_time_ms = start_event.elapsed_time(end_event)
    avg_gpu_time_ms = total_gpu_time_ms / iterations
    
    print(f"Average Kernel Execution Time: {avg_gpu_time_ms:.4f} ms")

    # Save results to a unique file
    result = {
        key: {
            'compile_time': compile_time,
            'avg_inference_time_ms': avg_gpu_time_ms,
            'total_inference_time_ms': total_gpu_time_ms
        }
    }
    with open(f"inductor_result_{H}_{W}_{kernel_size}.json", "w") as f:
        json.dump(result, f, indent=4)
