# Papers
* [Simple applications of BERT for Ad Hoc Document Retrieval, Yang et al., 2019](#simple-applications-of-bert-for-ad-hoc-document-retrieval)
* [End-to-End Open-Domain Question Answering with BERTserini, Yang et al., NAACL, 2019](#end-to-end-open-domain-question-answering-with-bertserini)


## A deep look into Neural Ranking models for IR 
Guo et al, 2019

**Gist**
* A survey paper on neural ranking models studying & analyzing the existing models w.r.t their assumptions, major design principles, and learning strategies.
* Includes comparison between representation focused vs interaction focused neural model architectures


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
