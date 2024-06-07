## Flash attention 2: Faster Attention with Better Parallelism and Work Partitioning
Tri Dao, July 2023

#### Gist – 
* This paper proposes FlashAttention 2 (extending FlashAttention 1), to further speed up FLOPs involved in Transformers attention computation
* Aims to further optimize attention computation which continues to be main bottleneck in scaling to longer sequences due to increased memory consumption

#### Approach –
* The current limitation in transformers architecture is identified to be due to inefficient work portioning between thread blocks and [warps of GPU](https://people.maths.ox.ac.uk/gilesm/old/pp10/lec2_2x2.pdf). A GPU warp is a group of 32 threads in a block of threads where all the threads in a warp execute the same task at the same time.
* This causes poor utilization of GPU and unnecessary read-writes in shared memory
* The approach optimizes work partitioning to tackle above limitations by-
*   Reducing the non-matmul FLOPs – since GPUs are built to optimally process matrix multiplication operations, non-matmul FLOPs take longer to process despite being a small fraction of the total FLOPs.
* Parallelizing both forward and backward pass along the sequence length dimension, in addition to batch and number of heads dimension (to increase GPU utilization)
* Performing work partitioning between different warps of thread blocks even for a single block of attention computation – to reduce communication & shared memory read/writes


Paper link - [FlashAttention2](https://tridao.me/publications/flash2/flash2.pdf)


## FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, June 2022
Dao et al, June 2022

#### Gist - 
* Propose an IO aware exact attention mechanism to speed up attention on hardware accelerators such as GPU
* Achieves it by reducing memory reads & writes between GPU’s fast on-chip SRAM and its relatively slower high bandwidth memory (HBM) using a technique called tiling
* Does not employ any approximation strategy

Tackles following limitations –

•	Due to higher compute speed than memory access, transformer models face bottleneck by memory access since their time and memory complexity is quadratic in sequence length, hindering the increase in context length as models and applications scale.

**What is tiling and how does it help?**
* To avoid reading and writing of attention matrix to and from HBM, this approach computes softmax reduction without accessing the entire input and also avoids storing the large intermediate attention matrix for backward pass.
* Above aspects are achieved by -
* Tiling - Restructuring the attention computation by splitting the input into blocks and making multiple passes over input blocks. This way it performs softmax reduction incrementally.
* Further, the approach involves storing the softmax normalization factor obtained during forward pass and uses it to quickly recompute the attention on-chip (SRAM) during backward pass. This step is faster than the standard approach of reading an intermediate attention matrix from HBM
* This is implemented in CUDA to gain fine-grained control over memory access & maintain all attention operations into a single GPU kernel.

#### Advantages –
* Despite the increase in FLOPs due to recomputation, this speeds up training compared to existing baselines and uses less memory (reducing its memory complexity from quadratic to linear)
* Enables longer context for sequence length yielding higher quality models

  Paper link - [FlashAttention, Dao et al, 2022](https://arxiv.org/pdf/2205.14135)
  
  Git repo - https://github.com/Dao-AILab/flash-attention 
