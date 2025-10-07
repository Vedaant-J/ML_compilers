import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel
import json
import time
import os

if __name__ == "__main__":
    torch.manual_seed(0)

    #Instantiate your PyTorch model
    
    N, C, H, W = 2, 3, 19, 19
    x = torch.randn(N, C, H, W).cuda()
    
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1).cuda().eval()

    # Torch-Inductor compilation
    scripted_model = torch.compile(model, backend="inductor")
    out = scripted_model(x)
    
    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("Inductor --- shape check:", out.shape == conv_ref.shape)
    print("Inductor --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))
    
    parameters_try = {
        'H_W': [8, 16, 32, 64, 128],
        'K_size':[3, 5, 7]
    }
    profile_results = json.load(open("inductor_profile_results1.json", "r")) if os.path.exists("inductor_profile_results1.json") else {}
    for H_W in parameters_try['H_W']:
        for K_size in parameters_try['K_size']:
            key = f"H/W:{H_W}_K:{K_size}"
            if key in profile_results:
                print(f"Skipping {key} as it's already profiled.")
                continue
            profile_results[key] = {}
            N, C, H, W = 1, 3, H_W, H_W
            out_channels=8
            kernel_size=K_size
            
            x = torch.randn(N, C, H, W).cuda()
            model = ConvModel(H, W, in_channels=C, out_channels=out_channels, kernel_size=kernel_size).cuda().eval()
            
            start_compile_time = time.time()
            compiled_model = torch.compile(model, backend="inductor")
            end_compile_time = time.time()
            compile_time = end_compile_time - start_compile_time
            _ = compiled_model(x) 
            torch.cuda.synchronize()
            actual_compile_time = time.time() - start_compile_time
            profile_results[key]['compile_time'] = compile_time
            profile_results[key]['actual_compile_time'] = actual_compile_time
            # with open(f"inductor_profile_results1.json", "w") as f:
            #     json.dump(profile_results, f)
            print(f"Inductor --- H/W: {H_W}, K: {K_size}, Compile time: {compile_time:.4f} seconds, Actual compile time: {actual_compile_time:.4f} seconds")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            iterations = 10
            start_event.record()
            for _ in range(iterations):
                _ = compiled_model(x)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            profile_results[key]['total_inference_time_ms'] = elapsed_time_ms
            profile_results[key]['avg_inference_time_ms'] = elapsed_time_ms / iterations
            # with open(f"inductor_profile_results1.json", "w") as f:
            #     json.dump(profile_results, f)
            print(f"Inductor --- H/W: {H_W}, K: {K_size}, Total inference time over {iterations} iterations: {elapsed_time_ms:.4f} ms, Average inference time: {elapsed_time_ms / iterations:.4f} ms")
            