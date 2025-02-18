# Papers

1. [On the Potential of Lexico-logical Alignments for Semantic Parsing to SQL Queries, Shi et al EMNLP 2020](#on-the-potential-of-lexico-logical-alignments-for-semantic-parsing-to-sql-queries)

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
