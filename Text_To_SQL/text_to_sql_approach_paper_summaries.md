# Papers
1. [RASL: Retrieval augmented schema linking for massive database Text-To-SQL, Eben et al., 2025](#rasl-retrieval-augmented-schema-linking-for-massive-database-text-to-sql)
2. [XiYan-SQL: A multi-generator ensemble framework for Text-To-SQL, Gao et al 2025](#xiyan-sql-a-multi-generator-ensemble-framework-for-text-to-sql)
3. [MAC-SQL: A multi-agent collaborative framework for Text-To-SQL, Wang et al, ACL 2025](#mac-sql-a-multi-agent-collaborative-framework-for-text-to-sql)
4. [Practiq - A Practical Conversational text-to-SQL dataset with Ambiguous and Unanswerable Queries, Dong et al., ACL 2025](#practiq-a-practical-conversational-text-to-sql-dataset-with-ambiguous-and-unanswerable-queries)
5. [CodeS: Towards Building Open-source Language Models for Text-to-SQL, Li et al, ACM 2024](#codes-towards-building-open-source-language-models-for-text-to-sql)
6. [Know what I don’t know – Handling ambiguous and unanswerable questions for text-to-sql, Wang et al., ACL 2023](#know-what-i-dont-know--handling-ambiguous-and-unanswerable-questions-for-text-to-sql)

## RASL: Retrieval augmented schema linking for massive database Text-To-SQL
Eben et al., 2025 (Amazon)

This is work in progress.

Paper link - https://arxiv.org/pdf/2507.23104


## XiYan-SQL: A multi-generator ensemble framework for Text-To-SQL

Gao et al, 2025

* Paper link - https://arxiv.org/pdf/2411.08599 
* Git repo - https://github.com/XGenerationLab/M-Schema 

**Gist –** 
* This paper presents a framework for Text-To-SQL called XiYAN-SQL based on ensemble of multiple generators for SQL candidates.
* Uses combined approaches of prompt engineering with In-context-learning (ICL) to maximize generation of high-quality & diverse SQL queries and SFT to achieve better controllability
* Proposes a two-stage and multi-task training strategy to train series of models with different preferences along with candidate selection strategy to select the most reasonable candidate
* Introduce M-schema – a semi-structured schema representation method designed to enhance the understanding of database structures
* The proposes framework – XiYAN-SQL achieves new SOTA execution accuracy of 75.63 % on Bird benchmark, 89% on SPIDER etc.
	
**Framework –**
* Introduces M-schema, based on MAC-SQL extended with additional information including column data types & description, primary key marking and value examples.
* Schema linking pipelines consists of retrieval module to search values and columns in database and column selector to remove irrelevant tables and columns post retrieval step.
   * First the LLM is prompted to identify keywords and entities in question, using which the retriever module retrieves top-k columns for each keyword.
   * Value retriever involves a two-phase retrieval strategy based on Locality Sensitive Hashing (LSH) and semantic similarity to identify similar values in the database. The final selected schema is the union set of column retriever and value retriever.
   * Column selector – The retrieved schema from previous step is presented to LLM organized M-schema along with few shot examples to evaluate the relevance of each column to user’s query and select only relevant ones.


**Candidate generation involves a -**
* Fine-tuned SQL generator obtained by 2 stage multi-task training approach (basic syntax training & generation enhance training) to fine-tune a series of high precision models with distinct advantages
* Different LLMs are used to rephrase the original query in multiple ways without altering its original meaning.
* ICL SQL generator –
   * The quality of SQL generation for a query also depends on the examples provided apart from the LLM’s reasoning capabilities.
   * Employs example selection strategy to include in the prompt to retrieve useful examples
* SQL refiner optimizes the generated SQLs obtained from previous step

**Candidate selection –** 
* Fine tunes a selection model to select the most reasonable candidate SQL among the various generated in previous step.
* This tackles the limitations of existing approaches which selects based on self-consistency which sometimes may not result in a consistent output at all or other times a consistent output itself could be incorrect too.

**Evaluation results & takeaways –**
* Evaluation datasets - BIRD, SPIDER, SQL-eval
* When used M-schema representation for 4 different LLMs – GPT 4o, Claude Sonnet 3.5, Gemini 1.5 Pro, all the models demonstrated improvements as compared using DDL schema representation in the prompt


## MAC-SQL: A Multi-agent collaborative framework for Text-To-SQL
Wang et al, ACL 2025

**Gist –**
* Introduces MAC-SQL, a multi agent collaborative framework for Text-To-SQL
* Achieves SOTA execution accuracy 59.59 (at time of writing paper) using GPT 4 as backbone with MAC-SQL on BIRD benchmark
* Also introduce instruction-tuning model – SQL-Llama for the task of Text-to-SQL

#### Components –
**Involves 3 agents -**
1. Selector –
   * When needed, it decomposes a large database into a smaller sub-database to minimize interference irrelevant information to LLM
   * Aim is to locate the minimal schema to answer the question with given knowledge
   * Motivation –
     * Introducing too many irrelevant schema items can result in LLM generating irrelevant schema items in the output SQL
     * Using complete database schema can lead to exceeding context length, adding to costs and issues from context length constraint
     * Activated only when the length of the database exceeds context-length, otherwise original schema is used
2. Decomposer agent -
  * This is the core agent for Text-to-SQL generation which uses selector and refiner on need basis
  * Decomposes complex original question into simpler sub-questions as reasoning steps & resolves them progressively by few shot CoT reasoning to get the final query
  * Aim is to enhance LLM’s reasoning ability by generating a series of intermediate steps (sub-questions and SQLs) before predicting the final SQL
  * First user’s question is judge for determining the complexity and if it can be answered by a single SQL query, the the query is generated directly, otherwise it decomposes it into sub-questions as mentioned above.

3. Refiner –
  * Performs SQL execution using tool and refines it if necessary in case of errors in generated SQL

**Evaluation -**
Benchmark datasets – BIRD, SPIDER

Paper link - https://aclanthology.org/2025.coling-main.36.pdf

## Practiq: A Practical Conversational text-to-SQL dataset with Ambiguous and Unanswerable Queries 
Dong et al., ACL 2025

Paper link - https://aclanthology.org/2025.naacl-long.13.pdf 

**Gist -**
* This paper studies existing real-world text-to-sql datasets
* Identifies 4 categories of ambiguous and 4 categories of unanswerable questions
* Implements framework to construct practical multi-turn conversational text-to-sql datasets consisting of ambiguous and unanswerable questions based on real-world datasets
* Multiple turns include – initial user question, assistant’s response seeking clarification, user’s clarification and assistant’s clarified SQL response with natural language explanation of the execution results
* To benchmark performance on this dataset, the paper implemented and evaluated SoTA LLM based baselines using different LLMs

**Definitions -**
* Ambiguous question – A question is ambiguous if it has multiple valid interpretations given the database schema
* Unanswerable question – A question is unanswerable if the corresponding database doesn’t contain the data that the question is asking for.
* Question categories –
    * Ambiguous question categories – ambiguous SELECT column, ambiguous Values within column, ambiguous WHERE column, ambiguous filter criteria
    * Unanswerable categories – Non-existent SELECT column, non-existent WHERE column, non-existent filter value, unsupported joins


**Approach for refining generated conversations and quality control –**
* A separate evaluation step is incorporated after each generation step to control the data quality besides optimizing the generation prompt
* This filtering step uses both LLM and execution checks
* LLM based approach –
    * Used to evaluate quality of the generated data from previous step
    * Rank different candidates if multiple candidates have been generated. E.g., For an ambiguous SELECT column in a questions, after generating multiple alternative columns for the SELECT column, a separate prompt with few shot examples is used for the LLM to evaluate whether the given multiple alternate columns are good candidates and make the question ambiguous.
    * For execution checks, in case of any Database change of generation of modified SQLs, the SQLs are executed against the database to ensure they are executable
    * Finally, after generating data for each category, an LLM is prompted to perform binary classification on whether the provided question and modified database pair belonged to the designed category or not. This classification is based on the definition of the category of ambiguous or unanswerable questions and severable human-curated examples. 


#### Dataset Evaluation & baseline –
2 evaluation tasks -

* Question category classification –
    * Classify whether the question is answerable or 1 of the 8 ambiguous/unanswerable categories
    * This is evaluated using classification accuracy to measure model’s classification performance
* Clarification SQL generation –
    * Predict the final SQL given the assistant’s clarification question and user’s clarification response
    * This is evaluated using execution accuracy to measure model’s performance


**Question category classification approach –**
* Employ few-shot prompting strategy for the question category classification task
* Experimented with different no. of few shot examples with different LLMs
* Prompt contains definition of each category along with a variable no. of examples per category
* Each example includes an input comprising the initial user question and relevant cell values retrieved via fuzzy matching approach ([Lin et al](https://aclanthology.org/2020.findings-emnlp.438.pdf)) & RAT-SQL ([Wang et al](https://aclanthology.org/2020.acl-main.677.pdf))
* The in-context demonstrations contain human-curated step-by-step thoughts and classification of the question categories

**Baseline experiments on handling ambiguities by SoTA LLMs -**
* Baseline approach – [DIN-SQL](https://proceedings.neurips.cc/paper_files/paper/2023/file/72223cc66f63ca1aa59edaec1b3670e6-Paper-Conference.pdf) – for predicting final clarification SQL
* Input to DIN-SQL framework – an ambiguous / unanswerable query without the assistant response or follow-up clarified user query
* As expected, model performs poorly as the framework is not designed to handle such queries
* For e.g., the model often hallucinates columns that do not exist in the database, potentially because the examples in the few-shot include only answerable questions

**Evaluated the dataset on 2 core tasks –**
* Question category classification
* SQL prediction


#### Results and error analysis –
**Takeaways -**
* Improving retrieval of cell values improves accuracy of identifying ambiguous / unanswerable question in text-to-sql system
* Ambiguous and unanswerable questions are challenging to handle even for SoTA LLMs in real-world practical Text-to-SQL data


**Observations on DIN-SQL performance on ambiguous and unanswerable queries –**
* For Ambiguous SELECT Column, for most cases, the generated SQL contains one of the ambiguous column names for the relevant entity in the user query. In few cases, the framework hallucinates, i.e., it assumes that entity mentioned in the user question is actually present as a column in the schema, in other cases the generated SQL doesn’t contain any of the related ambiguous columns from the schema for the entity in the user query.
* For ambiguous WHERE Column, all the predicted SQLs contain one of the ambiguous columns
* For Non-existent SELECT Column, (45%) cases lead to hallucination, i.e. the framework assumes that the entity in user query is present as a column in the schema and includes the column name in the final predicted SQL
* For Non-existent WHERE Column, only few cases lead to hallucination, the framework assumes that the entity in user query is present as column in the schema and the rest (78%) of the case are predicted as incorrect SQLs
* In the case of Unsupported Join we see that 56% of the SQLs are predicted with syntax errors/hallucinations where the framework assumes the presence of certain columns that do not exist in the schema to facilitate a JOIN operation to answer the question. Rest of the cases have logical errors, in the predicted SQL i.e., they contain JOIN columns that do not have any foreign key relationship
	

Git repo -
https://github.com/amazon-science/conversational-ambiguous-unanswerable-text2sql


## CodeS: Towards Building Open-source Language Models for Text-to-SQL
Li et al, ACM 2024

Paper link- https://arxiv.org/pdf/2402.16347

**Gist –**
* This paper introduces CodeS, a series of pre-trained open source language models (1B to 15 B parameters), specifically designed for Text-to-SQL task
* Studies research challenges in building CodeS including schema linking
* Evaluation on public benchmarks - BIRD, SPIDER and 2 real-world datasets for financial and academic application
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


## Know what I don’t know – Handling ambiguous and unanswerable questions for text-to-sql 
Wang et al., (Microsoft Asia) ACL 2023 

* Paper link - https://aclanthology.org/2023.findings-acl.352.pdf 
* Git repo – https://github.com/wbbeyourself/DTE 

**Gist –**
* This paper studies ambiguous and unanswerable cases in text-to-sql & summarizes them into 6 categories
* Identifies the causes behind each category and proposes requirements for handling ambiguous and unanswerable questions
* Proposes an effective approach to automatically produce ambiguous and unanswerable text-to-sql examples
* Proposes a weakly supervised DTE (Detecting then explaining) model for error detection, localization and explanation. 

**Definitions –**
* Ambiguous question - Questions that can have multiple semantic meanings based on a single table. E.g., the word “rating” in a user’s query could be mapped to disparate columns, such as “IMDB Rating”, “Rotten Tomatoes Rating”, or “Content Rating”.
   * Column ambiguity - Column ambiguity means that some tokens in the user question could be mapped to multiple columns.
   * Value ambiguity - means that some tokens in the user question could be mapped to multiple cell values in the table.
* Unanswerable questions – Questions that cannot be answered based on the information provided by the tables
   * Column unanswerable
   * value unanswerable
   * calculation unanswerable – This is more subtle. It requires mapping the concept mentioned in the user question to composite operations over existing table columns. E.g., For example, the balance of trade is a concept derived from “Exports − Imports”. Such mapping functions require external domain knowledge. Any model trained on general corpus, is likely to fail in such cases.
   * Out-of-scope – This category means that the question is out of SQL's operation scope, such as chart operations.

**Causes of ambiguous and unanswerable questions –**
* Unanswerable questions - end users are unfamiliar with the content of the table and don’t read the table carefully, 
* Ambiguity –
   * Arises due to the richness of natural language expressions and the habitual omission of expressions by users
   * Emergence of similar concepts in the table

**Expectations/requirements of an explainable parsing system –**
* To detect ambiguous and unanswerable questions
* To locate the specific reasons and generate corresponding explanations to guide the user in rectification.

**Motivation for handling such cases –**
* Enhances models’ ability to deal with problematic questions and improve user trust
* Makes it clear to users which part of their questions are problematic, guiding them to revise their questions


**Approach to handle ambiguous and unanswerable questions-**
* DTE model (Detecting then explaining)
   * This model is used to handle ambiguous and unanswerable questions
   * To locate ambiguous or unanswerable tokens in user query.
   * The location process is formulated as a sequence labelling problem, where each token in the user question will be tagged as being related to an ambiguous label, an unanswerable label, or others.
* DTE Has 3 modules -
   * Concept prediction module
   * Grounding module
   * Sequence labelling module
* Grounding module
   * Use ETA, a pre-trained probing based grounding model & modified the vanilla version to support handling of ambiguous and unanswerable questions
   * A heuristic-based baseline ([Sorokin and Gurevych, 2018](https://aclanthology.org/S18-2007.pdf)) is used in which n-gram matching via enumerating all n-gram (n < 5) phrases in natural language question and is done and which are linked to schema items via fuzzy string matching. A span is considered as ambiguous when it can fuzzy match with multiple results.
   * Similarly, if a noun phrase span can match no results, it is considered to be an unanswerable span.
 
     







