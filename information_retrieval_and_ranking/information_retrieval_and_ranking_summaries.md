# Papers

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
