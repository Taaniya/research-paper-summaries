# Papers
1. [CodeS: Towards Building Open-source Language Models for Text-to-SQL, ACM 2024](#codes-towards-building-open-source-language-models-for-text-to-sql)

## CodeS: Towards Building Open-source Language Models for Text-to-SQL
Li et al, ACM 2024

Paper link- https://arxiv.org/pdf/2402.16347

**Gist –**
* This paper introduces CodeS, a series of pre-trained open source language models (1B to 15 B parameters), specifically designed for Text-to-SQL task
* Studies research challenges in building CodeS including schema linking
* Evaluation on public benchmarks - BIRD, SPIDER and 2 real world datasets for financial and academic application
* Achieves SOTA accuracy and robustness on Text-to-SQL benchmarks

**Architecture –** 
* Built upon base model – StarCoder,  LLM designed specifically for code generation
* Significantly smaller than other LLMs like – GPT 3.5, GPT 4 and later.

#### Approach -
**What is schema linking –**
Defined as process to identify references database schemas (tables and columns) and database values within natural language questions.

**Introduce 3 components to solve Text-to-SQL problem-**
* Incremental pre-training (offline process) -
    *   Collect new corpus – 11 GB SQL-related data, 6 GB NL-to-code data, 4.5 GB NL related data
    *   Based on StarCoder, incremental pre-training is performed using SQL centric corpus to get CodeS
* Database prompt construction (Online process) –
    *   Schema filter – used to eliminate irrelevant tables & columns based on given query
    *   Value retriever – To extract potentially useful database values aligning with question
    *   Additionally, various metadata including datatypes, comments, representative column values, information on primary and foreign keys
* New domain adaptation (offline process) –
    *  Present a bi-directional data augmentation method to produce set of (question, SQL pairs) pairs for new domain databases.

**Schema filter –**
* Retains the most relevant tables and columns within the database for each question
* This includes a schema item classifier, trained to predict relevance scores for tables and columns according to user question.
* Using these scores, top k1 tables and for each table, top k2 columns are retained
* Using schema filter alleviates schema-linking pressure for the model during inference

**Value Retriever –**
* Retrieves values from Database that aligns with question, helping the Language model perform better schema linking.
* Proposes a coarse-to-fine matching approach to find matching values.
* Utilize BM25 index for a fast, coarse-grained initial search to get potentially relevant values, followed by a fine-grained matching processing using LCS (longest common substring) algorithm to calculate degree of match to find model relevant values. This BM25 index is built for all values in each database

**Database metadata –**
* To tackle ambiguities usually present in database schema such as similar column names etc, metadata including comments are included in the prompt to facilitate LLM in accurate schema linking.

**Representative database values –**
* 2 distinct values from each column are included in the prompt to inform the LLM the actual representations of values in each column. E.g., gender column has values as ‘M’  and ‘F’.

#### Evaluation –
**Benchmark Datasets-**
1.	[BIRD (larger dataset, more challenging)](https://proceedings.neurips.cc/paper_files/paper/2023/file/83fc8fab1710363050bbd1d4b8cc0021-Paper-Datasets_and_Benchmarks.pdf)
2.	SPIDER (contains 200 databases including 138 diverse domains)
3.	To assess robustness, 4 more challenging benchmarks-
a.	Spider-DK
b.	Spider-Syn
c.	Spider-Realistic
d.	Dr. Spider

**Evaluation metrics –**
* For Spider –
  * EX – Execution accuracy – evaluates whether the predicted and ground truth SQL queries yield the same execution results in database
  * Test-suite accuracy – evaluates whether the generated SQL query consistently passes the EX evaluation across multiple database instances
* BIRD –
  * Execution accuracy used as reliable and stable metric
* Schema item classifiers evaluated with AUC classifier metric

headers were kept as is for feature extraction. 

**Alignment annotation -**
* Annotators selected fragments from both sides – NLQ and SQL. The tokens need not be contiguous to be aligned. E.g., Non-contiguous ‘order by’ …. Limit 1 aligns with ‘the highest’ in NLQ.
* Overall 49% tokens in questions were aligned.
* Paper also shows distribution of tokens with POS tags which were aligned & which didn’t align with the SQL counterpart.
* Comparisons with other datasets –
  * WTQ includes more diverse semantics and logical operations
  * The family of SPIDER datasets contain queries even more complex than in WTQ, including a higher percentage of nested queries & multiple table joins.
  * Sequence labels for SQL type – column, literal (refer dataset row, column – nl_align), with this annotation

### Approach -
* Uses Seq2Seq as baseline – section 4
* Question & table encoding –
   * Uses 2 approaches – LSTM Bi-encoder & BERT as feature extractors
   * With BERT, the input is the concatenated sequence of question & column headers separated by [SEP] token. Each column is also separated by [SEP] token. The encodings are final layer representations
   * The NLQ and columns are passed as 2 segments to BERT encoder with segment ID set to 0 for all tokens of NLQ till [SEP] token between them & setting this ID to 1 from 1st token of column onwards.
   * Proposed approach in section 5
   * Code snippet –
      *  https://github.com/tzshi/squall/blob/main/model/model.py
      *  class TableParser. Method - _bert_features()
      *  TableParser is the encoder-decoder model
      *  Training logic is in main.py

**Using alignments in model training –**
* After feeding the concentrated forms of input questions & column headers, how do we teach the model to pay attention to column headers? – this is where alignments are used.
* The alignments provided, act as necessary supervision for the attention weights. This use of alignment instead of induced attention boosts the accuracy significantly.
* This alignment is incorporated during training as a finer-grained type of supervision improved results by 2.3 % by BERT and 3 % without BERT
* Loss function used in training strategy for alignment during model training –
* A Linear combination of loss terms of – seq2sql model, supervised attention, column prediction





