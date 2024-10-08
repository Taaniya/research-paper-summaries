# Papers

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
