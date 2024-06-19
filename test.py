import torch
import time

# Generate a large tensor
data = torch.randn(10000, 1000)

# Measure time to transfer to GPU
start_time = time.time()
data_gpu = data.to('cuda')
transfer_time = time.time() - start_time

# Directly create tensor on GPU
start_time = time.time()
data_gpu_direct = torch.randn(10000, 1000, device='cuda')
direct_load_time = time.time() - start_time

print(f"Transfer time: {transfer_time:.6f} seconds")
print(f"Direct load time: {direct_load_time:.6f} seconds")