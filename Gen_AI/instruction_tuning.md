# Papers

## The Natural Language Decathlon: Multitask Learning as Question Answering
McCann et al., ICLR 2019

1.	Framed 10 multiple NLP tasks as question answering. Leverages multi-task learning to improve zero-shot performance.
2.	One of the primary papers to inspire instruction tuning of Generative model for later papers - Prompt tuning, 2021, LaMDA, 2021, FLAN,2021, InstructGPT, 2023 etc.
3.	Aim – to obtain a general purpose model without making it train on task-specific data
5.	Lecture PPT (Stanford)- https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture17-multitask.pdf
6.	Training data size – 
7.	Custom architecture – 
-	1st baseline – pointer generator Sequence to sequence (S2S) model
-	Augmented S2S with self-attentive encoder & decoder layers
-	Add question pointer to previous baseline
-	QPtr enables coattended context and question – coattended question allows info from Q to directly flow into the decoder while generating answer
7.	Training Input – concatenation of context, question & answer
8.	Training objective – token level -ve log likelihood
9.	uses co-attention, self-attention & LSTM to create intermediate context state

**References-**
* Git repo - (Codebase – Code for data procuring, prep, training, evaluation available) - https://github.com/salesforce/decaNLP
* Paper link - https://openreview.net/pdf?id=B1lfHhR9tm


