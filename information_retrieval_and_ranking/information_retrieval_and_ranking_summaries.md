# Papers
* [A deep look into Neural Ranking models for Information Retrieval, Guo et al., 2019](#a-deep-look-into-neural-ranking-models-for-ir)
* [Simple applications of BERT for Ad Hoc Document Retrieval, Yang et al., 2019](#simple-applications-of-bert-for-ad-hoc-document-retrieval)
* [End-to-End Open-Domain Question Answering with BERTserini, Yang et al., NAACL, 2019](#end-to-end-open-domain-question-answering-with-bertserini)
* [A Deep Relevance Matching Model for Ad-hoc Retrieval, Guo et al, ACM 2017](#a-deep-relevance-matching)

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
