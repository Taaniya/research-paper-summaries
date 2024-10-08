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


## Learning to identify follow-up questions – 2020
Kundu et al., ACL, 2020

**Gist-**
* Introduce a follow up question identification task. Define the task in a conversational reading comprehension setting which supports automatic evaluation
* Present the dataset LIF – Learning to Identify Follow-up
* Approach – 3 way attentive pooling network to identify follow up given passage, conversation history & a candidate follow up question
* Makes the model to understand topic continuity and topic shift
* Binary classification Task - To predict whether a follow up question is valid or invalid

**Challenges in dataset-**
* The model is required to identify whether the subject of the question is the same as in the associated passage or in the conversation history, which is often distracted by the introduction of pronouns (e.g., I, he, she) and possessive pronouns (e.g., my, his, her). Such resolution of pronouns is a critical aspect while determining the validity of a follow-up question.
* It also needs to examine whether the actions and the characteristics of the subject described in the candidate follow-up question can be logically inferred from the associated passage or the conversation history.
* Moreover, capturing topic continuity and topic shift is necessary to determine the validity of a follow-up question.
* Subjects and their actions or characteristics in the invalid follow-up questions are often mentioned in the passages, but associated with different topics. 

**3 way attentive pooling approach-**
* Used to score each candidate follow up question
* The scoring is based on the relevance of candidate follow up question to conversation history considering 2 perspectives –
   - Considering associated passage
   - Without considering associated passage
   - Use binary cross entropy loss for training the model

Candidate scoring -
Score – f_sim(candidate , conversation_history | passage) + f_sim(candidate , conversation_history)  

Paper link - https://aclanthology.org/2020.acl-main.90.pdf



