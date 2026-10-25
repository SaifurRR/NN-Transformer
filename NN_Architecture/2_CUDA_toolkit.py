import time                                                  # for wall-clock timing
import torch                                                 # PyTorch
import torch.nn as nn                                        # layers

# ---------------------------------------------------------------- 1. pick a device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use the GPU if one is visible, else fall back to CPU
print("Using device:", device)                               # show which one was chosen
if device.type == "cuda":                                    # only query GPU details if we have one
    print("GPU name:", torch.cuda.get_device_name(0))        # name of GPU number 0

# ---------------------------------------------------------------- 2. tensors on the GPU
a = torch.randn(4096, 4096)                                  # random matrix created in CPU memory
b = torch.randn(4096, 4096)                                  # another one
a_gpu = a.to(device)                                         # copy to GPU memory (no-op if device is cpu)
b_gpu = b.to(device)                                         # same for b
c_gpu = a_gpu @ b_gpu                                        # matrix multiply runs on the GPU when the inputs live there
print("Result lives on:", c_gpu.device)                      # confirm where the result is
c_cpu = c_gpu.cpu()                                          # copy the result back to CPU memory (needed for numpy, printing, etc.)

# ---------------------------------------------------------------- 3. timing (GPU calls are asynchronous!)
def timed_matmul(x, y, repeats=10):                          # helper that averages several multiplies
    _ = x @ y                                                # warm-up run so one-time setup cost is not measured
    if x.is_cuda:                                            # only GPUs need explicit syncing
        torch.cuda.synchronize()                             # wait until the warm-up has actually finished
    start = time.perf_counter()                              # start the clock
    for _ in range(repeats):                                 # repeat to average out noise
        _ = x @ y                                            # the operation being timed
    if x.is_cuda:                                            # kernels are queued, not finished, when the call returns
        torch.cuda.synchronize()                             # wait for all queued work, otherwise the timing is too small
    return (time.perf_counter() - start) / repeats           # average seconds per multiply

print(f"CPU matmul: {timed_matmul(a, b) * 1000:.1f} ms")     # time on CPU
if device.type == "cuda":                                    # skip if there is no GPU
    print(f"GPU matmul: {timed_matmul(a_gpu, b_gpu) * 1000:.1f} ms")  # time on GPU

# ---------------------------------------------------------------- 4. training a model on the GPU
model = nn.Sequential(nn.Linear(128, 516), nn.ReLU(), nn.Linear(516, 1)).to(device)  # build the model, then move all its weights to the device
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)         # optimiser (created after .to(device), so it sees the GPU weights)
x = torch.randn(64, 128, device=device)                      # create the batch directly on the device (avoids a CPU->GPU copy)
y = torch.randn(64, 1, device=device)                        # targets must be on the same device as the model and inputs
for step in range(5):                                        # a few training steps
    loss = nn.functional.mse_loss(model(x), y)               # forward pass and loss, all computed on the GPU
    opt.zero_grad()                                          # clear old gradients
    loss.backward()                                          # backpropagation, also on the GPU
    opt.step()                                               # update weights
    print(f"step {step}: loss {loss.item():.4f}")            # .item() copies one number to the CPU so we can print it

# error encountered: "Expected all tensors to be on the same device"
# means the model and the data (or two tensors) are on different devices; call .to(device) on both.
