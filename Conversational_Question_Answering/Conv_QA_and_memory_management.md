# Papers
1. [Learning to Select the Relevant History Turns in Conversational Question Answering, Zaib et al., 2023](#learning-to-select-the-relevant-history-turns-in-conversational-question-answering)
2. [Keep Me Updated! Memory Management in Long-term Conversations, Bae et al., EMNLP 2022](#keep-me-updated-memory-management-in-long-term-conversations)

## Learning to Select the Relevant History Turns in Conversational Question Answering
 Zaib et al., 2023

**Gist-**
* Identify relevant history turns from conversation
* Identify relevant terms in the selected history turns with binary classification task
* Uses QR module – Question regeneration, given relevant conversation history turns and current question
* Demonstrates that dynamic history selection performs better then question rewriting
* Limitations trying to solve –
  - Current approach of rewriting question is performed without 

**Problem statement terms -**
* Conversational context – last question & answer turn
* The context entity refers to the entity mentioned from the conversational context
* Question entity is the entity targeted in the current question 

**Assumption –**
They assume the last turn as the context. Assuming that the incomplete question in current turn only need entities from last question to fill the missing pieces. 
i.e., even if the topic is returned to far back in history, the incompleteness in entities need only require last question to fulfill.


**Approach**
* 4 steps – Pruning, re-ranking, binary classification task
* Identify context entities and question entities using pre-trained BART. Input to model – current question & conversation history.
* Hard history selection - The turns that do not share similar context and question entities to the current question are pruned
* The remaining turns are then re-ranked on the basis of their relevance to the question. Their relevance is measured via the weightage assigned to them using the history attention mechanism 
* How to identify relevant history? - Turns that donot contain question or context entities

Paper link - https://arxiv.org/pdf/2308.02294

## Keep Me Updated! Memory Management in Long-term Conversations
 Bae et al., EMNLP, 2022

**Gist –**
* Formulate memory management task in long term conversations & construct its corresponding dataset
* Propose long term dialogue system including a novel memory management mechanism that selectively eliminates invalidated and redundant info
* Study methods of memorizing and updating dynamic information and utilizing them in successive dialogues
* Involves training to identify relevant memory sentences from history for effective retrieval


* Dialogue context – A sequence of chatbot and user utterances in a session. After each session, this is summarized into several sentences of user information.

**Long term dialogue system –**
Memory management system – decides which info to keep in memory 

**Memory management –** 
* Treats what operation to perform in memory as a classification problem – where given a pair of sentences, identifies whether the relationship is to ADD to memory, Replace in memory, drop something from memory or do nothing.

**Retriever –**
* Memory is represented in multiple memory sentences
* Memory retrieval is formulated as sentence retrieval problem by sentence encoder models to convert memory sentences into embeddings and storing them in a vector database
* At inference, top k memory sentences are retrieved using cosine similarity using search query
* Retriever model - Fine tune pre-trained BERT and use embeddings from 1st input token – [CLS]. Use triplet loss. Margin – 0.2
* Training settings – 20 epochs, 3hrs on 1 NVIDIA V100 GPU
* During inference, top k = 5 are retrieved using cosine similarity
  
**Evaluation-**
* Perform human evaluation to evaluate on following metrics -
  - Coherence, consistency, engagingness, humanness, memorability
* All above human evaluation metrics explained in appendix E

**References -**
* Dataset link – https://raw.githubusercontent.com/naver-ai/carecall-memory/master/data/carecall-memory_en_auto_translated.json
* Paper link - https://aclanthology.org/2022.findings-emnlp.276.pdf



