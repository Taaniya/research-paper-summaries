# Papers
1. [RoFormer: Enhanced transformers with Rotary Position Embeddings, Su et al., Nov 2023](#roformer-enhanced-transformers-with-rotary-position-embeddings)
2. [Flash Attention 2, 2023](#flash-attention-2-faster-attention-with-better-parallelism-and-work-partitioning)
3. [Flash Attention 1, 2022](#flashattention-fast-and-memory-efficient-exact-attention-with-io-awareness)
4. [LoRA, 2021](#lora-low-rank-adaptation-of-large-language-models--2021)
5. [BERT, 2018](#bert-pre-training-of-bidirectional-transformers-for-language-understanding)
6. [Distilling knowledge in neural network – 2015](#distilling-knowledge-in-neural-network--2015)

## RoFormer: Enhanced transformers with Rotary Position Embeddings
Su et al., Nov 2023

#### Gist -
* It combines the advantages of absolute and relative position encoding
* Encodes absolute position with a rotation matrix and incorporates relative position information in self-attention formulation
* Motivation to use position encodings in transformer model – Self-attention mechanism or current PLMs (Pre-trained Language Models) is position agnostic
* Different approaches have been proposed before to encode this position information in the learning process

#### Existing approaches for encoding position information –
* Absolute position encoding -
    * Using pre-defined function in original transformers architecture. This encoding is directly added to the contextual representations
    * Trainable absolute position encoding (as used in BERT, GPT, GPT 2 etc.)
    * Limitation –
        * Doesn’t generalize well due to not being flexible to context length. E.g., if initialized with a matrix of 512 x 768 dimensions for a context of 512 tokens and 768 dim, this matrix is updated during the training process. This makes it incapable of processing any sentences longer than 512 tokens. There is however discussion around overcoming this limitation with initializing position encodings longer than 512 and fine-tuning them. (reference - https://kexue.fm/archives/8130 , https://kexue.fm/archives/7947 )
* Relative position encoding – where relative position information is encoded in the attention mechanism
* ….other variants Etc.

#### Limitation of existing approaches –
•	Basically, all existing approaches commonly add the position information to the contextual representation which makes it unsuitable for linear self-attention architecture

### Background From original transformers paper-
* For a sequence of n tokens, with corresponding word embedding $E_N = {x_i}^N$, where $x_i \in R^d$ is the d-dimensional word embedding vector $w_i$ at position $i$ without the position information. The self-attention first incorporates the position information to the word embeddings and transforms them into query, key & value representations.
  
  $q_m = f_q(x_m,m)$

  $k_n = f_k(x_n, n)$

  $v_n = f_v(x_n, n)$

  * where, $q_m, k_n$ and $v_n$ incorporate the $m^{th}$ and $n^{th}$ positions through $f_q, f_k$ and $f_v$, respectively. The query and key vectors are used to compute attention scores while the weighted sum of value vectors is used to compute the output.

**Absolute Position Embedding** -
The function $f$ to incorporate absolute position embedding in self-attention mechanism for a vector $x_i$ at position $i$ is defined as -

$f_{t:t \in {q,k,v}} (x_i, i) := W_{t:t \in {q,k,v}} (x_i + p_i)$

* Where, $p_i \in R^d$ is a d-dimensional position embedding vector depending upon position of vector $x_i$.
* In the original transformers paper, the position embedding $p_i$ is generated using the sinusoidal function

  $p_{i, 2t} = sin(k/10000^{2t/d})$
  
  $p_{i, 2+t} = cos(k/10000^{2t/d})$

* where $p_{i, 2t}$ is the $2^{th}$ element of the d-dimensional position embedding vector $p_i$.

**Relative Position embedding**-

For relative position embeddings, the function to incorporate them in self-attention mechanism is defined as -

$f_q(x_m) = W_qx_m$

$f_k(x_n, n) = W_k(x_i + p_r^k)$

$f_v(x_n, n) = W_v(x_i + p_r^v)$

* Where, $p_r^k$ and $p_r^v$ are trainable relative position embeddings
* $r$ represents the relative distance between positions $m$ and $n$


#### References -
* https://huggingface.co/blog/designing-positional-encoding -
* https://youtu.be/o29P0Kpobz0?si=5PB40YR1Dp4gKMm- - Rotary position embedding
* https://youtu.be/EZufiIwwqFA?si=vjlitIE2gwMdpIhD  - Rotation matrix
* https://www.youtube.com/watch?v=n1vWKSSw4Uk&t=6s – Complex numbers and rotation matrix
* https://www.youtube.com/watch?v=kst2Io91JbM – Complex numbers as matrices
* https://www.youtube.com/watch?v=2CZhtdzUAi8&t=1s – Visualizing multiplication by i in complex plane
* https://www.youtube.com/watch?v=EZufiIwwqFA – Rotation matrix derivation
* https://www.youtube.com/watch?v=_zusa5ik_2g&t=1s – How to write complex number in polar form
* https://www.youtube.com/watch?v=Ty-4FnfY5i8 - Polar and Euler forms of a complex number

Paper link - https://arxiv.org/pdf/2104.09864


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

A matrix is low-rank if it has many fewer linearly independent columns than columns. Such matrices can be efficiently represented using rank-factorizations, which can be used to perform various computations rapidly. Many matrices appearing in applications, which are not genuinely low-rank, can be well-approximated by low-rank matrices; the best possible such approximation is given by the truncated singular value decomposition.  

Source - [Big Ideas in Applied Math: Low-rank Matrices, 2021, by Ethan N. Epperly](https://www.ethanepperly.com/index.php/2021/10/26/big-ideas-in-applied-math-low-rank-matrices/)

**Low-rank approximation** - 
The problem of low-rank approximation of a matrix is usually studied as approximating a given matrix by a matrix of low rank so that the Frobenius norm of the error in the approximation is minimized. 
Src - https://www.cs.cmu.edu/afs/cs/user/dwoodruf/www/cgklpw17.pdf 

LoRA explained - https://medium.com/@Shrishml/lora-low-rank-adaptation-from-the-first-principle-7e1adec71541 

Paper link - https://arxiv.org/pdf/2106.09685


## BERT Pre-training of Bidirectional Transformers for Language Understanding
Devlin et al., 2018

**Gist-**
* Designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers
* Can be fine-tuned with just one additional output layer for different tasks – QA, language inference, without substantial architectural changes
* Alleviates unidirectional nature of other language models (e.g., GPT) by using MLM pre-training objective
* Pre-trained with unlabeled data on pre-training 2 tasks – MLM (Masked Language Modelling & Next Sentence Prediction)
* Fine-tuning – First initialized with pre-trained parameters which are subsequently fine-tuned using labeled data on supervised downstream tasks with minimal modification to overall architecture
* Learns to produce contextual token representations from bidirectional context

**Architecture –**
* Multilayer bidirectional transformer encoder with 2 variations – Base & Large
    * Base – 12 layers, 768 dimensions, 12 attention heads, total params – 110 M
    * Large – 24 layers, 1024 dimensions, 16 attention heads, total params – 340 M
* Vocab size – 3000 token vocabulary with Word piece tokenizer
* Uses bidirectional self-attention whereas GPT uses constrained self-attention (only attend to previous tokens)
* Inputs are provided as single token sequences. In case of 2 input sentences, their token sequences are packed together and distinguished with special token [SEP] between them
* Additionally, a learned embedding is added to every token to indicate whether a sentence belongs to sentence A or sentence B
* Special tokens –
    * [SEP] – token inserted between 2 sentences to distinguish sentences in input token sequence
    * [CLS] – 1st token of every sequence. The final hidden vector corresponding to this token is used as aggregated sequence representation for classification tasks
* Final embeddings for an input token sequence is a obtained by summing up 3 components – token embeddings, position embeddings and segment embeddings of the corresponding token

**Pre-training –** 
* 2 unsupervised tasks - Masked Language Modelling & Next Sentence Prediction
* Datasets – BookCorpus & English Wikipedia (only text passages, lists, tables & headers are ignored). It was critical to use a document level corpus than a shuffled sentence level corpus to extract long contiguous sequences.
* **MLM –**
    * Masks 15 % of tokens in each sequence at random, and the model predicts these masked tokens
    * Masking – out of 15% of token chosen at random, the masking is performed by replacing an i-th token with-
        * [MASK] token, 80% of times
        * A random token, 10% of times
        * The same token, 10% of times (i.e., it remains unchanged). The purpose is to bias the representations towards the actual word
    * Final hidden vectors corresponding to masked tokens are fed to output softmax over the vocabulary, as in a standard language model.
    * This final hidden vector of the i-th token is used to predict the original token with cross entropy loss.
    * In contrast to previous approaches using denoising auto encoders, only masked tokens are predicted instead of reconstructing the entire input.
    * To tackle the downside of masking, which creates a mismatch between pre-training and fine-tuning (as [MASK] tokens do not appear in fine-tuning data), masked tokens are not always replaced with [MASK] token and hence also replaced with other tokens at random or remain unreplaced.
    * **Advantage of masking -** since the transformer encoder doesn't know tokens in the input it will be asked to predict and which tokens will be replaced, it will be forced to maintain a distributional contextual representation of every token in the input. Also, since the random replacement only happens for 1.5% (10% of 15%) tokens, it doesn't harm the language understanding capability of the model. 

* **NSP -**
    * Pre-trained on binarized NSP to train the model to understand the sentence relationships because many downstream tasks require this understanding and is not captured with standard language modelling task
    * Preparation – For collecting sentence pairs A and B, 50% of times, sentence B truly follows sentence A, while 50% times, B is chosen at random.
    * The final hidden vector corresponding to [CLS] token is used for NSP

**Fine-tuning –**
* Task specific models are formed for fine-tuning on different tasks with minimal changes to architecture
* This is done by incorporating BERT with additional output layer so that minimal no. of params need to be trained from scratch
* Tasks –
    * token level tasks – e.g., sequence tagging, Question answering
    * Sentence level tasks – e.g., MNLI, MRPC etc.

* Based on evaluation results and ablation studies, BERT is found effective for both feature based and fine-tuning based approaches.

**Feature based approaches** - Where contextual representations are learnt during training, and during inferences, this fixed features (contextual representations) are extracted and integrated from pre-trained model and applied to any task specific architecture.

Paper link - https://arxiv.org/pdf/1810.04805

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

**More References**-
* https://keras.io/examples/vision/knowledge_distillation/
* https://keras.io/examples/keras_recipes/better_knowledge_distillation/
* https://colab.research.google.com/github/patrickphatnguyen/Knowledge-Distillation-Keras/blob/master/Knowledge_Distillation_Notebook.ipynb
* https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker
