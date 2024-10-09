# Papers
1. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5), Raffel et al., JMLR](#exploring-the-limits-of-transfer-learning-with-a-unified-text-to-text-transformer)
2. [The Natural Language Decathlon: Multitask Learning as Question Answering (decaNLP), McCann et al., ICLR 2019](#the-natural-language-decathlon-multitask-learning-as-question-answering)


## Finetuned Language Models are Zero-shot learners
Wei et al., ICLR 2022

**Gist-**
* FLAN – Fine-tune Language Net. Uses LaMDA-PT as base model for instruction tuning
* 137 B parameter model, fine-tuned on 60 NLP tasks
* Architecture – Decoder only
* Objective – Full language modelling / Causal language modelling
* Focuses on exploring zero-shot capabilities for instruction type prompts. Though similar to DecaNLP ,but differs since DecaNLP focuses on multi-task learning whereas FLAN focuses on zero-shot capabilities.
* Motivation is to improve LMs ability to respond to NLP instructions
* Hypothesis – Idea is by using supervision to teach an LM to perform tasks via instructions, will also make it more capable to follow instructions and will do so for unseen NLP tasks.

**Approach –**
* Instructions used are similar to QA based task formulation and aims to unify NLP tasks by casting them as QA over a context
* Describes method for instruction-tuning – a method where a model is fine-tuned on multiple tasks phrased as instructions
* Outperforms GPT-3 on zero-shot performance for 20 datasets out of 25 tested
* Source code to load fine-tuning dataset is on git repo - https://github.com/google-research/flan
* More effective on tasks naturally verbalized as instructions than those which are directly formulated as language modelling
* Proves that training with instructions is crucial for zero-shot performance on unseen tasks.
* Plus this improvement only emerges with model scale

**Limitations-**
* Costly to serve due to large scaled model – 137 B parameters
* Compute intensive & time-consuming process for instruction-fine-tune – 60 hrs on a TPU with 128 cores. Slows down prototyping & development work, hinders frequent iteration of model improvements


**References-**
* git repo - https://github.com/google-research/flan
* Paper link - https://arxiv.org/pdf/2109.01652

## Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
Raffel at el., JMLR 2019

* Introduces a unified framework. Converts all text-based language problems into text-to-text format
* For every task, model is fed text & asked to produce some text. This keeps the training objective consistent for both pre-training and fine-tuning.
* This unified framework allows to apply same model, objective, training procedure and decoding process across all tasks
* Dataset trained on – C4 corpus
* Follows previous work – McCann – DecaNLP closely, but differences are-
* For T5, they separately fine-tune model on each task & use short task prefixes instead of QA format, whereas DecaNLP aimed to make models capable of multi-tasking
* T5 focuses on transfer learning rather than zero-shot learning

**Training-**
* standard MLE with teacher forcing, cross entropy loss
* unsupervised objective – denoising / MLM objective
* Architecture during fine-tuning – adapter layers or gradual unfreezing

Drawback of using LM with Causal masking in text-to-text settings –
* Model’s representation of input sequence is unnecessarily limited to causal masking of all tokens in the input sequence after a token i

**Model architectures considered –**
* Encoder-decoder –
* Decoder only with next word prediction autoregressively with causal attention mask
* Decoder only with prefix language mask for attention

**Results takeaways –**
* Encoder-decoder architecture with modified denoising / MLM pre-training objective performs the best compared to decoder only architecture with same Param count and computations operations
* Overall – MLM / Bert style / denoising pretraining performed the best, then prefix-LM
* Though also concludes that increasing model size and training time improves model performance, but also emphasizes that consideration for the eventual use of model is important when choosing between scaling methods
Note: Can visit Reflection section (section 4) in paper for more details
	
**References-**
* Paper link - https://jmlr.org/papers/volume21/20-074/20-074.pdf
* T5 model card on huggingface – https://huggingface.co/docs/transformers/model_doc/t5
* Code - https://github.com/google-research/text-to-text-transfer-transformer 

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


