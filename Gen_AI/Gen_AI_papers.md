# Papers
1. [CHIQ: Contextual History Enhancement for Improving Query Rewriting in Conversational Search, June 2024](#chiq-contextual-history-enhancement-for-improving-query-rewriting-in-conversational-search)
2. [ToolLLM: Facilitating LLMs to use 1600 real-world APIs, ICLR 2024](#toolllm-facilitating-llms-to-use-1600-real-world-apis-iclr-2024)
3. [What are tools anyway? A survey from Language Model Perspective, March 2024](#what-are-tools-anyway-a-survey-from-language-model-perspective)
4. [Toolformer, NeurIPS 2023](#toolformer-2023)
5. [Can You Unpack That? Learning to Rewrite Questions-in-Context, EMNLP 2019](#can-you-unpack-that-learning-to-rewrite-questions-in-context)
6. [GPT2 - Language Models are Multitask Learners, 2019](#gpt2---language-models-are-unsupervised-multitask-learners)
7. [GPT - Improving Language Understanding by Genrative Pre-training, 2018](#improving-language-understanding-by-generative-pre-training)


## CHIQ: Contextual History Enhancement for Improving Query Rewriting in Conversational Search
Mo et al., - June 2024
#### Gist –
 * This paper proposes a 2-step method to enhance quality of contextual history to improve query rewriting
 * Based on using open source LLMs and leveraging their capabilities to resolve ambiguities in conversation history before query rewriting
 * Demonstrate competitive results in comparison to systems with close source LLMs
 * LLM used – LLaMa-2 7B
 * Aim is to provide a clearer and less noisy version of history to be used instead of directly using it to generate/rewrite a search query.

#### Methods proposed for history enhancement –
 * Question disambiguation – Tackle ambiguous words, acronyms and coreference substitutes by prompting LLM to generate self-contained & unambiguous version of new user question based on conversation history
 * Response expansion – Enrich LLM’s previous response to make it self-contained using conversation history to achieve better retrieval of search query in subsequent steps.
 * Pseudo response -
 * Topic switch – LLM prompt to identify when topic switches between a user question and history and accordingly only include relevant parts of history. If topic is continued, conversation history is included as usual. If topic changes, only the last conversation is included to provide conversation and other irrelevant past chats are excluded from history to avoid LLM from rewriting query in subsequent steps.
 * History summary - Generate summary of history with relevant context only

#### Conversational search benchmark datasets used In experiments –
 * TopiOCQA – focuses on challenge of topic switch (https://aclanthology.org/2022.tacl-1.27.pdf )
 * QReCC – focuses on query rewriting – (https://aclanthology.org/2021.naacl-main.44.pdf )
 * CAsT – only used as test set - https://www.cs.cmu.edu/afs/cs.cmu.edu/Web/People/callan/Papers/trec2021-dalton.pdf

#### Evaluation –
 * Evaluation metrics -MRR, NDCG@3, Recall@10
 * Compared with systems –
   *   Traditional systems that fine tune small LMs for CQR (e.g., T5-base)
   *   systems that fine-tune an LLM-based retriever
   *   systems that directly obtain the rewritten query by prompting LLMs
  
Paper link - [CHIQ: Contextual History Enhancement for Improving Query Rewriting
in Conversational Search](https://arxiv.org/pdf/2406.05013)

## What are tools anyway? A survey from Language Model Perspective 
Wang et al, March 2024

#### Gist 
* Provide a unified definition of tools as external programs used by LLMs across a broad range of scenarios
* Perform a systematic review of LM tooling scenarios & approaches
* Analyze cost efficiency of tooling methods – give practical guidance on when & how one can use tools
* Offer concrete suggestions for evaluations

#### Limitations of LLMs that tools aim to solve –
* LLMs struggle to perform tasks that require complex skills – math, complex reasoning
* Fundamentally unable to solve tasks that require access to facts or information not included in their training data (e.g., current weather, latest events, current date etc)

Tools have been adopted to solve above limitations by facilitating a LM with capabilities it lacks.

##### What is a tool?
**Definition 1** - A computer program, specifically a function that can be applied to other objects and return an output.
Definition 1 – An LM used tool is a function interface to a computer program that runs externally to the LM, where the LM generates the function calls & input arguments to use that tool.

**Tools help in 3 major ways** – 
* Through perception – provide information collected from the environment. E.g., get_time()
* Performing action – exert action in the environment & change its state. E.g., make_post() request call to update website state
* Perform computation – perform computational tasks. These tasks can be beyond mathematical calculation e.g., language translator.

Many tools can fall into multiple categories as well. E.g., search engine can perform both perception & computation. For e.g.,  perceives environment – fetches data, retrieves documents & perform document similarity based search & choose relevant ones.

According to [Norvig & Peter (2010)](https://people.engr.tamu.edu/guni/csce421/files/AI_Russell_Norvig.pdf), **agents are defined as** anything that can be viewed as perceiving its environment through sensors and acting upon that environment through actuators.

Paper link - [What are tools anyway](https://zorazrw.github.io/files/WhatAreToolsAnyway.pdf)

## ToolLLM: Facilitating LLMs to use 1600 real world APIs, ICLR 2024
Qin et al

### Gist – 
* This paper presents a general tool-use framework ToolLLM that includes dataset construction, model training & evaluation
* Presents ToolBench – instruction tuning dataset for tool use. This is prepared automatically using ChatGPT
* Developed ToolEval – Automatic evaluator
* ToolLLama – This model is obtained by fine-tuning Llama on ToolBench dataset

### Motivation – 
To tackle limitations of current open source LLMs in tool-use capabilities since the existing instruction set are focused more on language tasks rather than tool-use domain

### Approach –
**3 stages of dataset construction** –
* APIs collection – collect APIs from RapidAPI hub
* Instruction generation – Prompting ChatGPT to generate diverse instructions involving these APIs, covering both single tool and multi-tool scenarios
* Solution path annotation – 
  o	Using ChatGPT to search for valid solution path 
  o	Implemented with a novel depth-first search-based decision tree (DFSDT) algorithm
  o	In contrast to CoT & ReACT, which explore only 1 possible direction & may suffer from error propagation, DFSDT enhances reasoning capabilities by enabling LLM to evaluate multiple reasoning traces and expand the search space and make deliberate decisions to either back track or proceed with an ongoing path


Git repo – https://github.com/OpenBMB/ToolBench  

Paper link – https://arxiv.org/pdf/2307.16789

Tool Eval - https://github.com/OpenBMB/ToolBench/tree/master/toolbench/tooleval

ToolLama - https://huggingface.co/ToolBench/ToolLLaMA-7b-v1 


## Toolformer, 2023
Schick et al, NeurIPS 2023

### Paper gist – 
* Introduces Toolformer Model -  finetuned to call and use externals APIs on different tasks
* Paper aims to show that models can teach themselves how to use external tools via API i.e., learn to decide when to call which API and how to incorporate results in next word prediction
* Model training is performed in self-supervised way using only a finite set of examples, without requiring human annotation
* Tools explored in this paper - Calculator, calendar, Q&A system, search engine
* Results highlight – improves zero-shot performance of different tasks


### Approach detail – 
* Base model used – GPT J (6.7 B params)
* Approach is agnostic of dataset used
* Each API call is represented as a tuple – a_c, i_c , where a_c – API call name, i_c are input parameter
* Both input & output of API are represented as text sequences

**Training dataset generation**-
* Dataset used – CCNet Wenzek et al. This dataset is annotated using GPT J model to incorporate API calls within it.
* Heuristics are used during annotation to sample only those API calls to incorporate in the dataset whose response are more likely to help the model predict the next token. This is done based on whether the API response tokens reduce the perplexity of future tokens
* As the generated dataset is an augmented version of original CCNet dataset, it has exact content as the original one, which helps the model learn to decide when & how to use which tool, based on its own feedback 

**Mode finetuning** –
* The annotated dataset is used to finetuning the model on standard LM objective (CLM)
* As the format of C* (annotated dataset) & original C dataset, the model learns to decide when & how to use which API call / tool

### Inference -
* Performs regular decoding until the model generates ‘->’ token indicating that model expects the API call response before generating next token. So, the relevant API call is made, and the response is fed back to the LLM followed by `</API>` special token.
* Subsequently, decoding is resumed.

### Experiment –
* Performed under zero-shot setting (no in-context learning examples provided)
* Standard greedy decoding process adopted with slight variation where the API call is generated in the output not only when `<API>` special token is most likely, but even if it’s among top K most likely tokens
* Evaluation performed on WikiText & held out CCNet dataset to monitor perplexity of different approach variations
* Experimented with 4 tools – calculator, search engine, Q & A, Calendar, Machine translation

### Result takeaways –
* API calls donot seem to be much useful to smaller models (GPT2 family), while the larger models are capable of learning how to make good use of them
* Self-supervised approach enabled the model to learn when & how to use a tool without requiring prompt demonstrating any task specific examples for the same

***Limitations*** – 
* Cannot be used in a chain (where O/P of one tool can be used as I/P for another one)
* Doesn’t let the LLM to use a tool in an interactive way

Paper link - [Toolformer, 2023](https://arxiv.org/pdf/2302.04761)

## Can You Unpack That? Learning to Rewrite Questions-in-Context
Elgohary et al., EMNLP 2019

* This paper introduces the task of query rewriting in a conversational context, where given a conversation history in context, the model generates a context- independent, self-contained question.
* The paper also releases a dataset – CANARD with 40,527 questions based on QUAC (Question Answering in Context) and trains seq2seq models on this for rewriting task
* The paper specifies the instructions given to crowd workers and the quality control mechanism followed during data collection. The efficacy of quality control is performed manually.

#### Dataset details -
* Dataset is constructed using human crowd workers tasked with making previously context-dependent questions to unambiguously answerable. This resolved coreference linkages.
* This dataset has multiple turns with variable turn lengths
* Data is created from QUAC [(Choi et al., 2018)](https://aclanthology.org/D18-1241.pdf), a conversational reading comprehension dataset 


**Baseline Model training data preparation –**
* The input sentence is created by concatenating all utterances in history H, prepending them to q_m and adding a special separator token between utterances.

**Evaluation –**
* Involves comparing BLEU score of baseline models with that of human rewrites


**Scripts –**
* https://github.com/aagohary/canard

**References -**
* Paper - https://aclanthology.org/D19-1605.pdf
* https://sites.google.com/view/qanta/projects/canard 


## GPT2 - Language Models are Unsupervised Multitask Learners
Radford et al., 2019

**Gist -**
* This paper continues the trend of more general methods of transfer of learned representations with pretraining phase. This work shows language models begin to learn multiple NLP tasks (Question answering, machine translation, summarization etc.) without any explicit supervision when trained on new dataset of millions of webpages – WebText
* The paper demonstrates that LMs can perform down stream tasks in few shot settings without any parameter or architecture modification
* Preliminary experiments confirmed that sufficiently large LMs (LLMs) are able to perform multi task learning in this toyish setup
* The capacity of the LLM is essential to the success of zero-shot task transfer and performance increasing this capacity improves its performance in log-linear fashion across tasks
* Training dataset – WebText (Millions of web pages)
* Largest Language Model GPT 2 is 1.5B parameters
* GPT 2 achieves SOTA results on 7 out of 8 language modelling tasks and still underfits the dataset

**Motivation –**
* Prevalence of single task training on single domain datasets majorly contributes to lack of generalization in diverse systems
* Multi-task learning () is a promising framework to improve general performance. This demonstrates language model performs downstream tasks in zero-shot setting.
* From meta learning perspective, each (dataset, objective) pair is a single training example sampled from distribution of datasets and objectives – implying that current systems would require 100s to 1000s of examples to induce functions to generalize well with multitask training which will be difficult to scale w.r.t dataset creation and designing of objectives.
* Current best performing systems on language tasks utilize a combination of pre-training and supervised fine-tuning that helps in more flexible forms of transfer.
* Recent works also suggest that task specific architectures are no longer necessary and transferring many self-attention blocks is sufficient
* This paper combines the above works and demonstrates the capability of zero-shot performance on down stream tasks without any parameter or architecture changes.


**Approach –**
* Core approach – Language modelling (unsupervised training)
* Single language has a natural sequential ordering, it is common to factorize the joint probabilities over symbols as product of conditional probabilities

$P(x) = \prod_{i=1}^{n}(s_n | s1,...,s_{n-1})$
 
* Learning to perform a single task can be expressed in a probabilistic framework as estimation of conditional distribution p(output|input). Since a general system should be able to perform many tasks, even for the same input, it should condition not only on the input but also on the task to be performed – i.e., it should model p(output | input, task). This has been formalized in multitask and meta learning settings. Though, task conditioning has been implemented at architectural level, recent work [McCann et al.,](https://arxiv.org/pdf/1806.08730) has shown that language provides a flexible way to specify tasks, inputs and outputs, all as a sequence of symbols. E.g., For example, a translation training example can be written as the sequence (translate to french, english text, french text)
* Language modelling is also able to learn the tasks without explicit supervision of which symbols are the outputs to be predicted.
* Since the supervised objective is the same as unsupervised objective, but only evaluated on a subset of sequence, the global minimum of the unsupervised objective is also the global minimum of the supervised objective.
* Preliminary experiments show that sufficiently large LMs are able to perform multi task learning but learning is much slower than in explicitly supervised approaches.

**Training set –**
* Own scraping – WebText – all outbound links from Reddit, which received atleast 3 karma (ensuring only human curated / filtered web pages are scraped)
* ~ 8 Million docs, 40 GB
* Not used Common Crawl due to data quality issues (unintelligible)
* Wanted the dataset to be of diverse domain & no assumptions of tasks to be performed ahead of time
* Removed Wikipedia docs due to overlapping training data with evaluation tasks
  
**Input representation –**
* [BPE - Byte-pair-encoding (Sennrich et al., Sennrich, 2015)](https://arxiv.org/pdf/1508.07909): Middle ground between word level LM & character level
(https://huggingface.co/learn/llm-course/en/chapter6/5)
* This tokenization algorithm effectively interpolates between word level inputs for frequent symbol sequences and character
level inputs for infrequent symbol sequences
* Though this is often operated on Unicode points rather than byte sequences, this approach can lead to very large base vocabulary – 130k (which is much larger than usual 32K to 64K)
* Also directly applying BPE to byte sequence results in sub-optimal merges due to BPE using greedy frequency based heuristics to build the vocab. Hence, this work modified some merging rules to improve compression efficiency and ensuring minimal fragmentation of words across multi vocab tokens

**Model –**
* Largely follows the details of Open AI GPT model with a few modifications
* Vocab size – 50, 257
* Batch size – 512
* Context size – 1024 tokens

Paper link - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf


## Improving Language Understanding by Generative Pre-Training
Radford et al., 2018

Paper link - https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

**Gist -**
* Demonstrates that gains on multiple NLP tasks can be achieved by generative pre-training a single task-agnostic language model on diverse text corpus of unlabeled text – followed by discriminative fine-tuning on specific task in NLP
* Works with language modelling objective on unlabeled data
* Explores semi-supervised approach for language understanding (involving unsupervised pre-training)
* Unsupervised pre-training objective – Standard LM objective
* Training data (Unsupervised pre-training) -
    * Book corpus dataset – 7K unique unpublished books (fantasy, romance), crucially containing long texts to learn long range dependencies.

**Model architecture -**
* 12 layer decoder, masked self-attention heads
* 768 dimensions, 12 multi-headed self-attention heads
* BPE – Byte-pair encoding vocab with 40K merges
* Masked self-attention: where every token can only attend to previous tokens in the self-attention layers
* Uni-directional language model, left-to-right architecture

**Pre-training experiment set up –**
* Adam optimizer
* LR scheduling – increasing lr linearly from 0 for 1st 2K updates & then annealed to 0 using cosine schedule
* Trained for 100 epochs 64 batch size
* Regularization with layer normalization
* Context size – 512 tokens (contiguous sequence)
* Activation function – GELU (Gaussian Error Linear Unit)
* Used learned position encodings instead of sinusoidal
* Data cleaning - Used ftfy library to clean raw book corpus dataset text, standardize punctuation & whitespace and used spacy tokenizer

**Evaluation –** 
* 4 Tasks –
    * Natural Language Inference
    * Question Answering
    * Semantic similarity
    * Text classification


