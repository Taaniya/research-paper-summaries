# Papers
* [Flash Attention 2, 2023](#flash-attention-2faster-attention-with-better-parallelism-and-work-partitioning)
* [Flash Attention 1, 2022](#flashattention-fast-and-memory-efficient-exact-attention-with-io-awareness)
* [LoRA, 2021](#lora-low-rank-adaptation-of-large-language-models--2021)
* [Distilling knowledge in neural network – 2015](#distilling-knowledge-in-neural-network--2015)

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


## FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
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


## LoRA: Low rank adaptation of Large Language Models – 2021
Hu et al, 2021

#### Gist –
* This paper introduces LoRA - Low Rank Adaptation, a storage and compute efficient approach for adapting Large Language Models to new tasks.
* Approach involves freezing of pre-trained model weights and injecting trainable rank decomposition matrices into each layer of transformers architecture, significantly reducing trainable parameters for downstream tasks
* This approach performs on-par or better than fine-tuning in model quality – RoBERTa, DeBERTa, GPT2, GPT3.
* Paper also provides empirical investigation into rank-deficiency in Language model adaptation


#### Hypothesis & approach - 
* Dense layers in neural networks undergo matrix multiplication
* The weight matrices in the layers are typically represented as full-rank
* However, pre-trained models show that during adaptation to a specific task that they have a low ‘intrinsic dimension ‘ i.e., these learned over-parameterized models reside on low intrinsic dimensions & can still learn efficiently with a random projection in a smaller subspace - Li et al (https://openreview.net/pdf?id=ryup8-WCW) , Aghajanyan et al (https://aclanthology.org/2021.acl-long.568.pdf ).
* The paper hypothesize that the change in weights during model adaptation also has low ‘intrinsic rank’, thereby allowing training of some dense layers in a neural network indirectly by optimizing rank decomposition matrices of the dense layers’ change during adaptation instead, while keeping the pre-trained weights frozen.
* This is achieved by constraining the weight update by representing the weight matrix $W_0$ with a low rank decomposition $W_0 + \delta W = W_0 + BA $

Where $W_0  \in R^{d\times{k}}$ , $B \in R^{d\times{r}}$ , $A \in R^{r\times{k}}$ and the rank $r << min(d, k)$

* During training, $W_0$ is frozen and doesn’t receive any gradient updates, while A & B contain trainable parameters.

#### Applying LoRA to transformers – 
* Out of the 6 weight matrices in transformers, this paper focuses on 4 attention weight matrices – $W_q, W_k, W_v, W_o$ and keep the other 2 MLP weight matrices as frozen for the experiment.


#### Advantages - 
* Leads to efficient training and reduces hardware constraints especially due to reduction in memory and storage usage as it doesn’t involve calculating gradients or maintain optimizer states for most of the parameters (as they are frozen) and need only do so for optimization of injected smaller rank matrices
•	Observe VRAM usage by 2/3 for transformer using Adam optimizer

#### Some background  readings and references – 
**Intrinsic dimension** - 
The intrinsic dimension for a data set can be thought of as the number of variables needed in a minimal representation of the data. 

**Rank of a matrix** - 
In linear algebra, the rank of a matrix A is the dimension of the vector space generated (or spanned) by its columns. This corresponds to the maximal number of linearly independent columns of A. This, in turn, is identical to the dimension of the vector space spanned by its rows. 

A matrix is low-rank if it has many fewer linearly independent columns than columns. Such matrices can be efficiently represented using rank-factorizations, which can be used to perform various computations rapidly. Many matrices appearing in applications which are not genuinely low-rank can be well-approximated by low-rank matrices; the best possible such approximation is given by the truncated singular value decomposition.  

Src - https://www.ethanepperly.com/index.php/2021/10/26/big-ideas-in-applied-math-low-rank-matrices/#:~:text=Upshot%3A%20A%20matrix%20is%20low,to%20perform%20various%20computations%20rapidly. 

**Low-rank approximation** - 
The problem of low-rank approximation of a matrix is usually studied as approximating a given matrix by a matrix of low rank so that the Frobenius norm of the error in the approximation is minimized. 
Src - https://www.cs.cmu.edu/afs/cs/user/dwoodruf/www/cgklpw17.pdf 

LoRA explained - https://medium.com/@Shrishml/lora-low-rank-adaptation-from-the-first-principle-7e1adec71541 

Paper link - https://arxiv.org/pdf/2106.09685


## Distilling knowledge in neural network – 2015 

Hinton et al, 2015.

#### Gist-
* Usual training requires large language models to learn structures from huge data but deployment requires models with low latency & be relatively light weight to use relatively low computational resources
* This paper introduces a method referred as Distilling knowledge from an ensemble into a single model
* We can train a very large & cumbersome model if that makes it easier to extract structure from the data. And once it has been trained, 'distillation' training technique can be used to transfer knowledge from the cumbersome model to a small model that is more suitable for deployment

#### Approach -
In distillation, knowledge is transferred to the distilled model by training it on transfer set and using a soft target distribution for each case in the transfer set that is produced by the cumbersome model with a high temperature in its softmax. The same high temperature is used when training a distilled model, and once it is trained, the temperature is set back to 1.

* Training objective during knowledge distillation – Weighted average of 2 objective functions
* Cross entropy with soft targets distribution, while using the same high temperature used by teacher model
* Cross entropy with correct labels, using logits in softmax by student, but using temperature of 1

Paper link - https://arxiv.org/pdf/1503.02531
