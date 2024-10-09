# Papers
1. [Training language models to follow instructions with human feedback,  Ouyang et al, OpenAI, NeurIPS 2022](#training-language-models-to-follow-instructions-with-human-feedback)
2. [Fine-tuned Language Models are Zero-shot learners, Wei et al., ICLR 2022](#finetuned-language-models-are-zero-shot-learners)
3. [Multitask prompted training enables zero-shot task generalization, Sanh et al, ICLR 2022](#multitask-prompted-training-enables-zero-shot-task-generalization)
4. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5), Raffel et al., JMLR](#exploring-the-limits-of-transfer-learning-with-a-unified-text-to-text-transformer)
5. [The Natural Language Decathlon: Multitask Learning as Question Answering (decaNLP), McCann et al., ICLR 2019](#the-natural-language-decathlon-multitask-learning-as-question-answering)

## SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions
By Wang et al., ACL 2023

* Framework for instruction following capabilities of model
* Annotation-free method to align LMs on instruction following behavior
* Introduces a semi-automated process of instruction-tuning with model’s own instruction signals. Uses an iterative bootstrapping algorithm
* Also release synthetic dataset of 52K instructions & set of manually written tasks to build and evaluate instruction following models
* Evaluated on benchmark – [SUPER NATURAL INSTRUCTIONS dataset by Wang, 2022]](https://aclanthology.org/2022.emnlp-main.340.pdf)
* Base model – GPT-3 fine-tuned with these curated instructions dataset
* Performance results are close to [Instruct GPT](#training-language-models-to-follow-instructions-with-human-feedback)
* Work is similar to data augmentation, except that it is task-agnostic instruct generation. Motivation is to bootstrap new tasks

**Data generation step –**
* Model used to generate instances – GPT 3 – davinci – text-davinci-001
* Fine-tuning step
  - input – concatenate instruction & instance input
  - output – target
  - Set the prompt loss weight to 0
  - Fine-tuned model – GPT3 – davinci (same model used to generate instruction data)
* Fine-tuning performed using OpenAI’s fine-tuning API. Set the prompt loss weight to 0
* Git repo - https://github.com/yizhongw/self-instruct 


## Training language models to follow instructions with human feedback
By Ouyang et al, OpenAI, NeurIPS 2022

* Propose InstructGPT models
* Parameters – 1.3 B
* Source model – GPT3
* Fine-tuning twice for aligning model on user’s intent -
  - Demonstrated behaviour dataset created by labelers
  - on ranked model output dataset by human feedback. Labelers labeled their preferences over multiple outputs by a model for the same prompt.

* Applied Reinforcement learning to align with human values -
  - Train a reward model (RM) to predict which of the model output out of multiple outputs for every prompt, will the labeler prefer.
  - This RM is used to as a reward function to fine-tune supervised learning baseline to maximize this reward using PPO
  - In other words, this reward function models the preferences of human labelers & provides feedback to the RL agent. This agent is trained with PPO algorithm.

Paper link - https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf


## Finetuned Language Models are Zero-shot learners
Wei et al., ICLR 2022

**Gist -**
* FLAN – Fine-tune Language Net. Uses LaMDA-PT as base model for instruction tuning
* 137 B parameter model, fine-tuned on 60 NLP tasks
* Architecture – Decoder only
* Objective – Full language modelling / Causal language modelling
* Focuses on exploring zero-shot capabilities for instruction type prompts. Though similar to [DecaNLP, 2019](#the-natural-language-decathlon-multitask-learning-as-question-answering), but differs since DecaNLP focuses on multi-task learning whereas FLAN focuses on zero-shot capabilities.
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

**Limitations -**
* Costly to serve due to large scaled model – 137 B parameters
* Compute intensive & time-consuming process for instruction-fine-tune – 60 hrs on a TPU with 128 cores. Slows down prototyping & development work, hinders frequent iteration of model improvements


**References -**
* git repo - https://github.com/google-research/flan
* Paper link - https://arxiv.org/pdf/2109.01652


## Multitask prompted training enables zero-shot task generalization

By Sanh et al, ICLR 2022

**Gist -**
* Test question at scale - Can zero-shot generalization also be induced by explicit multi-task learning? – by developing a system to convert any natutal language task into human-readable prompt form - multi task prompt training
* Why explicit multi-task learning ? while LLMs are already hypothesized to generalize well to new tasks so far as a result of implicit process of multi-task learning
  - Because this ability requires sufficiently large model
  - And is sensitive to wording of its prompts
* Hence, paper focuses on – explicitly training LMs in a supervised and multitask fashion & aim to make model robust to wording choices of prompts

**Architecture –**
* T0 – An Encoder-decoder  (T5 variant). Lester’s LM adapted T5 model : T5 + LM 
* 11 B parameter version of T5+LM
* Previous works (T5 and others) of using NL to describe underlying NLP task use single QA prompt due to which model doesn’t generalize well to new prompts or new tasks which aren’t expressed in their fixed format
* Note: What's the difference between this work and FLAN although closely related? –
  - Prompts are more diverse than used in FLAN and due to encoder-decoder architecture with MLM pre-training objective, model performs better after multi-task prompt training whereas FLAN degrades.

**Approach -**
* Uses a training mixture containing large set of different tasks specific in NL prompts
* Training dataset – 62 datasets 12 tasks
* Develop a templating language & application to diverse datasets into prompts
* Templates are functions mapping a data example into natural language for the input and target sequences
* Each dataset has multiple prompt templates consisting of an input and a target template
* More details section 4 - Unified prompt format

**Creating prompts from raw datasets -**
* Crowdwork. Contributors created diverse, open prompts for each dataset
* Training method -
  - Fine-tuned T0 i.e.,T5+LM model on multi-task training mixture dataset of NL prompts mentioned above.
  - It is trained to generate output only. Like Encoder decoder model does.

**Results -**
* T0 (11B) better than baseline – T5+LM
* T0 (11B) matches or exceeds GPT-3 on 9 out of 11 tasks
* Also recorded results of smaller 3B version of T0. From the results in eppendix F, this version either performs at par with baseline T5+LM or exceeds it but still underperforms 11B T0 model
* For comparison purpose - GPT-3 is 175 B param

**Takeaways –**
* Training on more prompts per dataset leads to better and more robust generalization to held-out tasks
* Releases models and prompts used to fine-tune models

**References -**
* Git repo – how to create NL prompts for NLP tasks - https://github.com/bigscience-workshop/promptsource
* Paper link - https://openreview.net/pdf?id=9Vrb9D0WI4




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


