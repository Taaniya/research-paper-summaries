# Papers
1. [CHIQ: Contextual History Enhancement for Improving Query Rewriting in Conversational Search, June 2024](#chiq-contextual-history-enhancement-for-improving-query-rewriting-in-conversational-search)
2. [ToolLLM: Facilitating LLMs to use 1600 real-world APIs, ICLR 2024](#toolllm-facilitating-llms-to-use-1600-real-world-apis-iclr-2024)
3. [What are tools anyway? A survey from Language Model Perspective, March 2024](#what-are-tools-anyway-a-survey-from-language-model-perspective)
4. [Toolformer, NeurIPS 2023](#toolformer-2023)
5. [Can You Unpack That? Learning to Rewrite Questions-in-Context, EMNLP 2019](#can-you-unpack-that-learning-to-rewrite-questions-in-context)


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
