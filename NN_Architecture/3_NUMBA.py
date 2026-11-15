
import math
import time
import numpy as np
from numba import cuda, float32


@cuda.jit                                   
def vec_add(a, b, c):
    i = cuda.grid(1)                        
    if i < c.size:                          
        c[i] = a[i] + b[i]                 

@cuda.jit
def scale_and_clip(img, out, gain, lo, hi):
    row, col = cuda.grid(2)                 
    if row < img.shape[0] and col < img.shape[1]:
        v = img[row, col] * gain
        out[row, col] = min(max(v, lo), hi) 

TILE = 16     

@cuda.jit
def matmul_tiled(A, B, C):
    sA = cuda.shared.array((TILE, TILE), dtype=float32)   
    sB = cuda.shared.array((TILE, TILE), dtype=float32)

    row, col = cuda.grid(2)                
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    acc = float32(0.0)

    for t in range((A.shape[1] + TILE - 1) // TILE):      
        k = t * TILE + ty
        sA[tx, ty] = A[row, k] if row < A.shape[0] and k < A.shape[1] else 0.0
        k = t * TILE + tx
        sB[tx, ty] = B[k, col] if k < B.shape[0] and col < B.shape[1] else 0.0
        cuda.syncthreads()                  

        for j in range(TILE):               
            acc += sA[tx, j] * sB[j, ty]
        cuda.syncthreads()                  

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = acc


def main():
    simulated = cuda.is_available() is False or bool(int(__import__("os").environ.get("NUMBA_ENABLE_CUDASIM", "0")))
    n = 10_000 if simulated else 10_000_000  

    a = np.full(n, 1.0, dtype=np.float32)
    b = np.full(n, 2.0, dtype=np.float32)
    d_a = cuda.to_device(a)                 
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(a)         

    threads = 256
    blocks = math.ceil(n / threads)         
    vec_add[blocks, threads](d_a, d_b, d_c) 
    cuda.synchronize()                      
    c = d_c.copy_to_host()                  
    print("vec_add ok:", np.allclose(c, 3.0))

    img = np.random.rand(300, 500).astype(np.float32)
    d_img = cuda.to_device(img)
    d_out = cuda.device_array_like(img)
    tpb = (16, 16)                          
    bpg = (math.ceil(img.shape[0] / tpb[0]), math.ceil(img.shape[1] / tpb[1]))
    scale_and_clip[bpg, tpb](d_img, d_out, 1.5, 0.0, 1.0)
    print("scale_and_clip ok:", np.allclose(d_out.copy_to_host(), np.clip(img * 1.5, 0, 1)))

    m = 64 if simulated else 1024
    A = np.random.rand(m, m).astype(np.float32)
    B = np.random.rand(m, m).astype(np.float32)
    d_A, d_B = cuda.to_device(A), cuda.to_device(B)
    d_C = cuda.device_array((m, m), dtype=np.float32)
    bpg = (math.ceil(m / TILE), math.ceil(m / TILE))
    matmul_tiled[bpg, (TILE, TILE)](d_A, d_B, d_C)       
    cuda.synchronize()
    print("matmul ok:", np.allclose(d_C.copy_to_host(), A @ B, rtol=1e-3, atol=1e-3))

    if not simulated:                       
        start = time.perf_counter()
        for _ in range(10):
            matmul_tiled[bpg, (TILE, TILE)](d_A, d_B, d_C)
        cuda.synchronize()
        ms = (time.perf_counter() - start) / 10 * 1000
        print(f"matmul {m}x{m}: {ms:.2f} ms per call, {2 * m**3 / (ms / 1000) / 1e9:.1f} GFLOP/s")


if __name__ == "__main__":
    main()