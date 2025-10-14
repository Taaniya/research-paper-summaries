# Papers
1. [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (UnigramLM), ACL 2018](#subword-regularization-improving-neural-network-translation-models-with-multiple-subword-candidates)
2. [Neural Machine Translation of rare words with subword units (BPE), ACL 2016](#neural-machine-translation-of-rare-words-with-subword-units-2016) 
   


## Distributional Properties of Subword Regularization
Cognetta et al., EMNLP 2024

Summary yet to be added!

Paper link - https://aclanthology.org/2024.emnlp-main.600.pdf


## MaxMatch-Dropout: Subword Regularization for WordPiece
ACL, 2024

Summary yet to be added!

https://aclanthology.org/2022.coling-1.430.pdf


## Fast WordPiece tokenization
Song et al., ACL 2021 

Summary yet to be added!

Paper link - https://aclanthology.org/2021.emnlp-main.160.pdf


## BPE-dropout: Simple and Effective sub word regularization
ACL 2020

Summary yet to be added!

Paper link - https://aclanthology.org/2020.acl-main.170.pdf


## Sentence Piece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing
Kudo and Richardson, ACL 2018

Summary yet to be added!

Paper link - https://aclanthology.org/D18-2012.pdf)


## Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates

Taku Kudo, ACL 2018

**Gist -**
* This paper introduces the concept of sub word regularization
* It proposes regularization for open vocabulary for NMT (Neural Machine Translation) with no change in model architecture
* Also proposes a new subword segmentation algorithm based on unigram language model, which is capable of outputting multiple subword segmentations with probabilities (section 3 in paper)
* Central idea of subword regularization – inject noise to input sentences by randomly changing internal representations to increase robustness by virtually augmenting training data with on-the-fly subword sampling -> to improve the accuracy and robustness of the NMT models
* With sub word regularization, NMT model becomes robust and achieves higher accuracy

### What is sub word regularization –
**Background and limitation of existing sub word segmentation approach using BPE -**
* During tokenization, where text is converted to sub word units, sub word segmentation is ambiguous and multiple segment sequences are possible for a sentence with the same vocabulary. 
* Even though different subword sequences are possible for the same input, NMT handles each subword sequence as different inputs. This becomes more apparent while converting subword sequences into id sequences. These variants are kind of [spurious ambiguities](https://aclanthology.org/J18-2003.pdf ) which may not always be resolved in the decoding stage.
* Limitation with existing BPE segmentation algorithm is that BPE encodes a sentence into a unique subword sequence as it is based on a greedy and deterministic symbol replacement, and cannot provide multiple segmentations with probabilities. 

### Sub word regularization
* In this, multiple segmentation candidates during the training time, will make the model more robust to noise & segmentation errors, as they indirectly help the model learn the compositionality of words – e.g., books is composed of “book” + “s”.
* This will train the model on different inputs randomly sampled from the original input sequences. This paper regards this approach as a variant of ensemble training and drawing analogy with dropout regularization which is also a kind of ensemble training, where many different models are trained on different subsets of data (since dropout randomly switches off a subset of hidden units during training.
* This paper also draws analogy with data augmentation, as subword regularization involves converting input sentence into multiple invariants similar to data augmentation for image classification (image flipping, distortion etc)


**Unigram subword segmentation Model –**
* Capable of outputting multiple segmentation candidates with probabilities
* Assumes that each subword occurs independently and consequently the probability of subword sequence $x$ is the product of probabilities of each subword $p(x_i)$.

  $x = (x_1, x_2,..., x_M)$

  $P(x) = \prod_{i=1}^M p(x_i) $

  $\forall x_i \in V, \sum_{x\in V} p(x) = 1$

  * V - pre-determined vocabulary
* The most probable segmentation $x^*$ for an input sequence is given by -

  $x^* = argmax_{x \in S(x)}p(x)$
   * where, $S$ - set of segmentation candidates built from input sequence $X$.
   * $X$ -> source sequence for NMT
   * $Y$ -> target sequence for NMT
   * $x^*$ is obtained with viterbi algorithm 

* In real setting, vocab set is unknown and since joint optimization of vocab and their occurrence prob is intractable, it is found out through iterative algorithm.

**Subword sampling –** 
* Subword regularization samples one subword segmentation distribution P(x|X) for each parameter update.
* This paper then deep dives on how approximate sampling is performed by using l-best segmentations and l-best search using forward DP A* search algorithm.

**BPE vs Unigram model for subword sampling –**
* Though both BPE and unigram model encodes input sentence into fewer bits (subword segments) with a certain data compression principle (BPE – dictionary, unigram – entropy), the unigram model is more flexible w.r.t its utility in subword regularization as it is based on probabilistic language model and can output multiple segments with their probabilities.


Paper link - https://aclanthology.org/P18-1007.pdf


## Neural Machine Translation of rare words with subword units, 2016

Sennrich et al., 2016

Original paper link - https://aclanthology.org/P16-1162.pdf

**Gist -**
* This paper introduces a variant of Byte Pair Encoding for tokenization (word segmentation) as a capable mechanism for encoding open vocabularies in the context of Neural Machine translation application
* This approach results in a compact vocabulary of variable-length sub word units
* This achieves higher accuracy in translation of rare and unseen words as a sequence of sub word units than existing baselines based on back-off dictionary
* This paper also discusses other word segmentation techniques including character n-gram models

**Intuition –**
* Various word classes are translatable via smaller units than words, for instance names (via character copying or transliteration), compounds (via compositional translation), and cognates and loanwords (via phonological and morphological transformations).

**Motivation –** 
* Translation of some words is transparent / obvious even if some words are unknown, based on a translation of known sub word units such as morphemes or phonemes. Word categories whose translation is potentially transparent include:
    * named entities - Between languages that share an alphabet, names can often be copied from source to target text. Transcription or transliteration may be required, especially if the alphabets or syllabaries differ
    * Cognates and loanwords - Cognates and loanwords with a common origin can differ in regular ways between languages, so that character-level translation rules are sufficient. Example - claustrophobia (English), Klaustrophobie (German)
    * Morphologically complex words - Words containing multiple morphemes, for instance formed via compounding, affixation, or inflection, may be translatable by translating the morphemes separately. Example: solar system (English), Sonnensystem (Sonne + System) (German)

### Byte Pair Encoding (BPE) – 
* Byte Pair Encoding (BPE), basically is a simple data compression technique that iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte.
* This paper adapts this algorithm for word segmentation. Instead of merging frequent pairs of bytes, the approach in this paper merges characters or character sequences.

#### Approach –
* First, the symbol vocabulary is initialized with character vocabulary, and each word is represented as a sequence of characters, plus a special end-of-word symbol ‘.’. This special token allows restoring of the original tokenization after translation.
* Then iteratively, all symbol pairs are counted and each occurrence of the most frequent pair (‘A’, ‘B’) is replaced with their new symbol ‘AB’. This is called merging.
* Each merge operation results in a new symbol which represents a character ngram.
* Frequent character n-grams (or whole words) are eventually merged into a single symbol
* The final symbol vocabulary size is equal to the size of the initial vocabulary, plus the no. of merge operations
* **No. of merge operations** is the only hyperparameter in this algorithm
* Major advantage of sub-word units is that the symbol sequences are still interpretable as sub-word units, and that the neural network can generalize to translate and produce new words (unseen at the time) on the basis of these sub-word units
* BPE segmentation requires $O(N^2)$ computational cost when it naively scans the pair of symbols in every iteration (Ref - [SentencePiece](https://aclanthology.org/D18-2012.pdf))

### Applications
* BPE tokenization is used in GPT family of models
* Llama uses a SentencePiece BPE tokenizer
* Mistral models use byte-fallback BPE tokenizer
  
