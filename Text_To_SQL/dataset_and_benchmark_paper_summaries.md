# Papers

1. [BIRD: Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs, Li et al, NeurIPS, 2023](#)
2. [On the Potential of Lexico-logical Alignments for Semantic Parsing to SQL Queries, Shi et al EMNLP 2020](#on-the-potential-of-lexico-logical-alignments-for-semantic-parsing-to-sql-queries)

## On the Potential of Lexico-logical Alignments for Semantic Parsing to SQL Queries
Shi et al, 2020 (Training & evaluation Dataset)

* Paper link - https://aclanthology.org/2020.findings-emnlp.167.pdf
* SQUALL Dataset link – https://github.com/tzshi/squall
* Git repo – https://github.com/tzshi/squall
* Dataset format - https://github.com/tzshi/squall?tab=readme-ov-file#squall-dataset-format 

**Gist-**
* Solves previous challenge of just having logical forms for NL input as training set. Since just having logical forms do not indicate important fine-grained relationships between individual words & logical form tokens
* Release dataset built on top of WikiTableQuestionss (WTQ) dataset & enriches 11,276 questions of WikiSQL questions with manually created SQL equivalents between SQL and question fragments
* Proposes and proves that richer supervision will help more in semantic parsing of NLQ to generate SQL for Text-to-SQL
* Proposes 2 methods –
  * Supervised attention
  * Using lexical alignments during model training
  * Adopting an auxiliary objective of disambiguating references in input queries to table columns also called column prediction – to infer column in data table a question fragment refers to.
* Dataset contribution –
   *  Train – 9.03 K tokens
   *  dev – 2.24 K tokens
   *  test – 4.34 K tokens
* Dataset annotations including labelling columns & literal values (Filter values)
* WikiTables has around 11.2 K tables with 22k questions, out of which SQUALL has 11.2 K questions.
* WikiTables is the base dataset chosen having QA pairs where the answer is present in semi-structured form in table
* This paper contributes to creating SQL queries for these questions & also align question tokens with corresponding SQL query fragments
* They suffixed every column name with their data type
* For annotation consistency, all tables are assigned the same name & columns with sequential names – c1, c2…. In database schema. But the original table

## Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs 
Li et al. NeurIPS , 2023

**Gist** -  
* Present a new text-to-SQL benchmark dataset that better represent real life scenarios
* Paper emphasizes the database values
* Highlights challenges of text-to-SQL problem on real world databases
* Emphasizes these challenges on big databases
* Demonstrate significance of database values through experimental results of models struggling to perform on BIRD as compared to human performance baseline on metrics - execution accuracy
* Models evaluated – GPT 3.5 Turbo, Claude 2, GPT 4-32 K
* Also proposes a new evaluation metric – VES (Value efficiency score) to evaluate efficiency of generated SQLs.

**Limitations of previous benchmarks –**
* Focus on database schema with few rows of DB values – very different from real world scenario where dataset sizes are large and have often noisy values and require a thorough understanding of database values

### Dataset features –
* Contains 12,751 text-to-SQL pairs, 95 databases spanning over 33 GB & 37 professional domains
* dirty and noisy database values
* external knowledge grounding between natural questions and database values
* SQL efficiency
* Domains include – blockchain, sports, health care, politics etc.

### External knowledge evidence for question annotation -
Based on the findings conclude that external knowledge evidence is required to map natural language instructions into counterpart database values. These evidences are categorised into 4 –
 * Numeric reasoning - mathematical computation required for certain SQL operations e.g., MINUS, ADDITION, DIVISION, MULTIPLY, percentages, formulas etc.
 * Domain knowledge - domain specific knowledge used to generate SQL operations e.g., calculating return on investment, next income etc.
 * Synonym knowledge – includes words / expressions that have same meaning
 * Value illustrations- refers to detailed descriptions of database values, including value types, value categories, mapping combinations of columns and values that correspond to entities. E.g., gender can be represented as ‘M’ and ‘F’ in gender column in customer database
* Database values bring more challenges in text-to-SQLs. Questions are classified into 2 macro categories –
  * Fundamental type – refer to those questions which can be answered without database value comprehension. E.g., match bases, ranking, aggregation, counting etc.
  * Reasoning type – Questions that demand external knowledge grounding on values. E.g., question about domain knowledge, numeric computing, synonym, value illustration 

### Evaluation- 
**Metrics –** 
* Execution accuracy - proportion of examples in the evaluation set for which the executed results of both the predicted and ground-truth SQLs are identical, relative to the overall number of SQLs
* Valid efficiency score - measure the efficiency of valid SQLs generated by models

### Experiments and results analysis –
* The incorporation of a dedicated reasoning prompt by DIN-SQL, DIN-SQL + GPT-4 to achieve a new state-of-the-art result on BIRD. This contains value sampling, few shot demonstrations and self correction
* **Knowledge evidence analysis –**
  * After being easily fed with the external knowledge evidence about the database values, all models have a clear improvement across the different difficulty levels. This indicates that external knowledge evidence in BIRD is effective and instructive for models to better understand the database values
  * ICL-based approaches have a better self-knowledge grounding capability and pre-trained SQL knowledge than FT smaller models with less than 5B parameters.
  * Equipped with COT, ChatGPT can perform better, since multi-step reasoning is beneficial when the knowledge and data are low-resource

### Error analysis –
Error categories-
**Wrong schema linking –**
* ICL-based approaches have a better self-knowledge grounding capability and pre-trained SQL knowledge than FT smaller models with less than 5B parameters.
* Equipped with COT, ChatGPT can perform better, since multi-step reasoning since beneficial when the knowledge and data are low-resource scenario where ChatGPT can accurately comprehend the structure of the database but erroneously associates it with inappropriate columns and tables.
* **Misunderstanding database content -**
  * Occurs when ChatGPT either fails to recall the correct database structure or generates fake schema items
  * Making LLMs understand the database structured and values is till a pain point
* **Misunderstanding knowledge evidence –**
  * Refers to cases in which the model does not accurately interpret human-annotated evidence.
  * This indicates that ChatGPT exhibits a lack of robustness in response to unfamiliar prompts or knowledge, causing it to directly replicate formulas without considering SQL syntax

Paper link - https://proceedings.neurips.cc/paper_files/paper/2023/file/83fc8fab1710363050bbd1d4b8cc0021-Paper-Datasets_and_Benchmarks.pdf

