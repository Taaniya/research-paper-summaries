## FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, June 2022
Dao et al, June 2022

#### Gist - 
* Propose an IO aware exact attention mechanism to speed up attention on hardware accelerators such as GPU
* Achieves it by reducing memory reads & writes between GPU’s fast on-chip SRAM and its relatively slower high bandwidth memory (HBM) using a technique called tiling

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
