# Papers
1. [Sentence-T5 (ST5): Scalable Sentence Encoders from Pre-trained Text-to-Text Models, Ni etal., ACL, 2022](#sentence-t5-st5-scalable-sentence-encoders-from-pre-trained-text-to-text-models)
2. [Text Embeddings by Weakly-Supervised Contrastive Pre-training, Wang et al., 2022](#efficiently-teaching-an-effective-dense-retriever-with-balanced-topic-aware-sampling)
3. [Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling, Hofstatter et al, SIGIR, 2021](#efficiently-teaching-an-effective-dense-retriever-with-balanced-topic-aware-sampling)
4. [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT, Khattab & Zaharia, SIGIR 2020](#colbert-efficient-and-effective-passage-search-via-contextualized-late-interaction-over-bert)
5. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, Reimers and Gurevych, EMNLP 2019](#sentence-bert-sentence-embeddings-using-siamese-bert-networks)
6. [A deep look into Neural Ranking models for Information Retrieval, Guo et al., 2019](#a-deep-look-into-neural-ranking-models-for-ir)
7. [Simple applications of BERT for Ad Hoc Document Retrieval, Yang et al., 2019](#simple-applications-of-bert-for-ad-hoc-document-retrieval)
8. [End-to-End Open-Domain Question Answering with BERTserini, Yang et al., NAACL, 2019](#end-to-end-open-domain-question-answering-with-bertserini)
9. [A Deep Relevance Matching Model for Ad-hoc Retrieval, Guo et al, ACM 2017](#a-deep-relevance-matching-model-for-ad-hoc-retrieval)

## Sentence-T5 (ST5): Scalable Sentence Encoders from Pre-trained Text-to-Text Models
Ni et al., ACL 2022

* Aim - Introduces Sentence T5 Google, explores use of T5 for sentence embeddings
* Explore 3 variations of sentence T5 models / turning a T5 model into a sentence embedding model-
  - ST5 first: use 1st token representation of encoder output
  - ST5 mean: use avg of all token representations of encoder output
  - Encoder-Decoder first : use 1st token representation from decoder output, when input text is fed to the encoder and only ‘start’ symbol is fed to the decoder as input

**Training –**
* Sentence encoder training using dual encoder architectures
* Once both the modules sharing the weights create their representation of the input text respectively, projection and L2 normalization is applied to the resulting embeddings.
* Next the embeddings from both the towers are scored for similarity with dot product. Since L2 norm is applied the similarity score is cosine similarity
* 2-stage training -
  - Training on web mined conversational input-response & question answering pairs
  - Contrastive learning on NLI pairs

**References-**
* Model Card - https://huggingface.co/sentence-transformers/sentence-t5-base 
* Paper link - https://aclanthology.org/2022.findings-acl.146.pdf 


## Text Embeddings by Weakly-Supervised Contrastive Pre-training
Wang et al., 2022

* This paper introduces E5, text embeddings model
* Training data e.g., [MSMARCO](https://arxiv.org/pdf/1611.09268), NQ, NLI
* Data prep –
  - Curate a text pair dataset CCPairs (colossal clean text pairs) by harvesting heterogenuous semi structured data
  - Strategy to choose negative samples – in-batch negatives
  - Dataset used for general training (using contrastive training) –
* Fine-tuning with labeled data – NLI (STS & linear probing tasks), MS-MARCO passage ranking dataset, NQ (Natural Questions) (MARCO and NQ for retrieval tasks)
* Training method – contrastive training for initial training of the pre=trained encoder model, followed by further training on labelled dataModel architecture
* Transformer encoder – bert-base-uncased, bert-large-uncased-word-masking
* Evaluation – method, dataset, metrics
  - Methods –
    * BEIR – metrics – nDCG@10
    * MTEB - accuracy, v-measure, average precision, MAP, nDCG@10, and Spearman coefficients

Paper link - https://arxiv.org/pdf/2212.03533.pdf

## Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling
Hofstätter et al., SIGIR 2021

**Gist –** 
* Proposes TAS Balanced – Topic aware sampling technique to compose training batches for training a dense passage retriever model
* Paper contribution involves improving both pairwise and in-batch teacher signals
* Topics are clustered before training, using semantic dot product similarity
* Trained on 400K queries from MSMARCO dataset
* Queries are sampled & selected from a single cluster to concentrate information about a topic in a single batch, which after in-batch negative teaching, leads to higher quality results
* To be used for asymmetric search
* Distilbert for dense passage retrieval with balanced topic-aware sampling
* Optimized for the task of semantic search – Mentioned in huggingface model card
* Suitable for dot product distance metric

**Models -** 
* Main dense retrieval model – dual encoder BERTDOT
* 2 teacher architectures –
  -  For combination of pairwise BERTCAT
  -  In batch negative teaching ColBERT
* Evaluation dataset - TREC

**References -**
* https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b
* Paper link - https://arxiv.org/pdf/2104.06967.pdf
* https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval
* Example notebook using the model for query and passage embeddings
https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval/blob/main/minimal_bert_dot_usage_example.ipynb 


## ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
Omar Khattab and Matei Zaharia, SIGIR 2020

**Gist -**
* Introduces ranking model adapting BERT for efficient retrieval, that uses contextualized late interactions over BERT
* Introduces late interaction architecture that independently encodes query and document using BERT to get their respective contextualized embeddings. This interaction step that models their fine-grained similarity. This is followed by computing their relevance using cheap and pruning-friendly approach that enables faster computations without exhaustively evaluating every candidate.
* This architecture’s effectiveness is competitive with BERT based models & outperforms non-BERT based approaches & significantly demonstrates faster performance compared to the latter.
* Training & Evaluation datasets – passage search, MS MARCO & TREC CAR.

#### Approach –
* The approach basically combines fine-grained matching of interaction-based models and pre-computation in representation-based models and yet strategically delays the query-document interaction in the overall execution flow.
* Fixed size embedding dimensions -

**Query encoding -**
* Query is encoded by appending token [Q] immediately after prepending [CSL] token in the beginning of given query, tokenizing the sequence using WordPiece upto pre-defined no. of tokens $N_q$ (by either padding it with special [MASK] token or truncating them) and subsequently passing them through BERT encoder to get contextualized embeddings for each token in the sequence.
* Then the embeddings are passed through a linear layer without any activations, but producing output of m-dimensions, where m is fixed to a smaller size than BERT’s hidden size dimensions.
* Paper also provides reasons for making above decisions based on observations of various ablation studies.

**Document encoding –**
* Similarly, each document is encoded by first tokenizing it, adding token [D] after prepending [CLS] token in the beginning.

**Late interaction approach to compute relevance score –**
* Dot product of bag of contextualized embeddings of all query tokens – $E_q$ and contextualized embeddings of each token in document d, $E_d$.
* To compute score for each document, its reduced across all document tokens with max_pool to choose document term with maximum similarity. Similarly, this token embeddings similarity with each query token embeddings is summed up to compute the relevance score. All the documents are sorted by this relevance score for ranking.

**Indexing technique –**
* To enable end-to-end retrieval with ranking for a large collection, this is achieved by applying MaxSim operator for each query embedding across all documents in the collection using fast vector-similarity data structures to conduct this search efficiently using FAISS implementation of an efficient index – IVFPQ (Inverted File with Product Quantization).
* This index partitions the embeddings into a pre-defined number of cells which helps in faster search by searching in only nearest partitions at runtime. To optimize memory efficiency, the embeddings are divided into sub-vectors such that similarity computations are performed within this compressed domain leading to cheaper computations and faster performance.


**Model training –**
* BERT is fine-tuned, while additional parameters for linear layer and embeddings for [Q] & [D] are trained from scratch using Adam optimizer

Paper link - https://arxiv.org/pdf/2004.12832
Git repo - https://github.com/stanford-futuredata/ColBERT

## Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
Nils Reimers and Iryna Gurevych, EMNLP 2019

**Gist -**
* Introduces a modification to BERT network architecture and is computationally efficient
* The modification is addition of pooling operation to output of BERT.
* To fine-tune BERT, the architecture is also modified to use Siamese and triplet network structures to derive semantically meaningful sentence embeddings
* Purpose is to tackle computational overhead on STS (semantic text similarity)
* This way the new architecture enables BERT to be applicable for new tasks –
* Large scale semantic similarity comparison, clustering, information retrieval via semantic-search
* SBERT fine-tunes BERT in a Siamese / triplet network architecture i.e., in this paper, they started with a pre-trained BERT model & fine-tuned it to yield useful sentence embeddings

#### SBERT Model architecture
* With SBERT modified architecture, the addition is a pooling operation to the output of BERT / RoBERTa to derive a fixed size sentence embedding
* Explore 3 pooling strategies –
  - Using output of CLS token
  - Computing mean of all output vectors (default)
  - Computing max-over-time

#### Evaluation -
* Experimented with 2 set ups -
  - Directly fine-tuned on STS benchmark data
  - Pre-trained BERT on NLI, then fine-tuned on STS benchmark dataset. This shows better results.

Paper link - https://aclanthology.org/D19-1410.pdf 


## A deep look into Neural Ranking models for IR 
Guo et al, 2019

**Gist**
* A survey paper on neural ranking models studying & analyzing the existing models w.r.t their assumptions, major design principles, and learning strategies.
* Includes comparison between representation focused vs interaction focused neural model architectures

Paper link - https://ciir-publications.cs.umass.edu/getpdf.php?id=1407

## Simple applications of BERT for Ad Hoc Document Retrieval

Yang et al., 2019

* Test their hypothesis that BERT can be finetuned to capture document relevance when applied on ad hoc document retrieval task
* Also describe approach to handle scoring document relevance when doc length exceeds BERT’s max input length
* Approach is built on top of previous approach - [BERTserini](#end-to-end-open-domain-question-answering-with-bertserini)

**Approach –**
* Finetunes on another dataset with short sentence-level annotations of Microblog social media posts on document retrieval task
* Also finetuned on QA task but of similar domain as that of test data – News wire.
* Flow includes 2 steps – retrieval using BM25 (Anserini Approach) and feeding top retrieved doc to BERT classifier. BERT score is combined with retrieved document through linear interpolation
* Input to BERT classifier is a concatenation of query and retrieved document - [[CLS], Q, [SEP], D,[SEP]], and classifier output is obtained by using [CLS] token vector as input to a single layer neural network to classify whether its relevant.
* Handles scoring long documents by performing classification inference over each sentence in retrieved document, selecting the sentence with highest score, and combining it with original doc score with linear interpolation.

**Takeaways –**
•	Finetuning on document retrieval task contributes better to it learning & better relevance scoring performance rather than on training on similar domain but of different task.

Paper link - https://arxiv.org/pdf/1903.10972

## End-to-End Open-Domain Question Answering with BERTserini

Yang et al., NAACL 2019

**Gist** 
* Applies BERT reader to score candidate retrieved answers for a Question answering system
* Finetuned BERT on SQUAD dataset to score relevant answers for a question

**Approach –**
1.	Question answering is performed in 2 stages – retrieving top k documents with BM25 and subsequently passing retrieved docs to BERT reader to score each doc. Finally BERT score is combined with document retrieval score using linear interpolation
2.	Experiment with different granularity (chunking strategy for indexing passages – sentences, passage, article level).
3.	BERT is used to score each passage without using its last softmax layer across all answer spans.

**Takeaways –** 
* Paragraph level retrieval of answer document works best among the 3 granularities experimented.
* BERT reader needs improvement w.r.t weighted interpolation between BERT and retrieval scores.

Paper link - https://aclanthology.org/N19-4013.pdf

### A Deep Relevance Matching Model for Ad-hoc Retrieval
by Guo et al, ACM 2017

* This paper identifies the ad-hoc retrieval task as relevance matching problem in information retrieval rather than semantic matching problem and highlights the difference between the two problems.
* Proposes deep neural relevance matching model – DRMM for ad-hoc retrieval


* Matching problem is defined as -
  
$Match(T1, T2) – F(\phi(T_{1}), \phi(T_{2}))$

$\phi$ – function to map each text to a representation vector

$F$ – scoring function based on their interactions between them

Depending upon the choice of these 2 functions, there are 2 types of deep matching models based on architectures –

**Representation focused –**
   * Build a good representation for a single text with a DNN. Then conducts matching between compositional and abstract text representations.
   * $\phi$ is a complex representation of mapping function, while $F$ is relatively simple matching function. Eg., in [DSSM](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf), $\phi$ is a Feed Forward NN, while $F$ is a cosine similarity function.
   * Without loss of generality, all the model architectures of representation-focused models can be viewed as a Siamese (symmetric) architecture.

**Interaction-focused model –**
   * First builds local interactions (i.e., local matching signals) between two pieces of text, and then uses deep neural networks to learn hierarchical interaction patterns for matching. 
   * $\phi$ is a simple mapping function, while F is a complex deep model. E.g., in [DeepMatch](https://papers.nips.cc/paper_files/paper/2013/file/8a0e1141fd37fa5b98d5bb769ba1a7cc-Paper.pdf), $\phi$ maps each text to a sequence of words, $F$ is a Feed Forward NN powered by a topic model over word interactions matrix

Differences in semantic matching and relevance matching problem –

**Semantic matching -**
*   Semantic matching problem can come in different forms in NLP tasks such as -paraphrase identification, question answering, automatic conversation, as these involve identifying semantic relations between 2 pieces of text.
*   Semantic matching assumes that the 2 pieces of text are homogeneous and consist of a few natural language sentences like – Question/Answers, sentences, dialogs etc.

This basically emphasizes on –
* Similarity matching signals - to capture the semantic similarity/relatedness between words, phrases and sentences, as compared with exact matching signals.
* Compositional meanings - to use the compositional meaning of the sentences in natural language based on grammatical structures rather than treating them as a set/sequence of words
* Global matching requirement - Semantic matching usually treats the two pieces of text as a whole to infer the semantic relations between them, leading to a global matching requirement

**Relevance matching –**
*   The matching in ad-hoc retrieval, is about relevance matching, i.e., identifying whether a document is relevant to a given query, where a query is typically short and keyword based while document can be of varying lengths.

Following factors are considered to estimate relevance between a query and a document.
* Exact matching signals: the exact matching of terms in documents with those in queries is the most important signal in ad-hoc retrieval due to the indexing and search paradigm in modern search engines.
* Query term importance: Since queries are mainly short & keyword based without complex grammatical structures, in ad-hoc retrieval it should be taken into account.
* Diverse matching requirement: in ad-hoc retrieval, a relevant document length can be very long and relevance matching could happen in any part of the relevant document and we do not require the document as a whole to be relevant to a query.


Proposed DRMM model –
* Interaction focused-model employing a joint deep architecture at query term level for relevance matching
* First builds local interactions between each pair of terms from a query and a document based on term embeddings.
* For each query term, the variable-length local interactions are mapped into a fixed-length matching histogram.
* Based on this fixed-length matching histogram, a feed forward matching network is employed to learn hierarchical matching patterns and produce a matching score.
* Finally, the overall matching score is computed by aggregating the scores from each query term with
a term gating network computing the aggregation weights.

Paper link - https://ciir-publications.cs.umass.edu/getpdf.php?id=1247
