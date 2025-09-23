# Papers
1. [Neural Machine Translation of rare words with subword units, ACL 2016](#neural-machine-translation-of-rare-words-with-subword-units-2016) 

## Neural Machine Translation of rare words with subword units, 2016

Sennrich et al., 2016

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

### Byte Pair Encoding (BPE) –** 
* Byte Pair Encoding (BPE), basically is a simple data compression technique that iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte.
* This paper adapts this algorithm for word segmentation. Instead of merging frequent pairs of bytes, the approach in this paper merges characters or character sequences.

#### Approach –
* First, the symbol vocabulary is initialized with character vocabulary, and each word is represented as a sequence of characters, plus a special end-of-word symbol ‘.’. This special token allows restoring of the original tokenization after translation.
* Then iteratively, all symbol pairs are counted and each occurrence of the most frequent pair (‘A’, ‘B’) is replaced with their new symbol ‘AB’. This is called merging.
* Each merge operation results in a new symbol which represents a character ngram.
* Frequent character n-grams (or whole words) are eventually merged into a single symbol
* The final symbol vocabulary size is equal to the size of the initial vocabulary, plus the no. of merge operations
* **No. of merge operations is the only hyperparameter in this algorithm**
* Major advantage of sub word units is that the symbol sequences are still interpretable as sub word units, and that the neural network can generalize to translate and produce new words (unseen at the time) on the basis of these sub word units

Paper link - https://aclanthology.org/P16-1162.pdf
