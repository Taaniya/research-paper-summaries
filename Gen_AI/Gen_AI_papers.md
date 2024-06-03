## What are tools anyway? A survey from Language Model Perspective (WIP)
Wang et al, 2024

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

## Toolformer, 2023
Schick et al

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
* Performs regular decoding until the model generates ‘->’ token indicating that model expects the API call response before generating next token. So, the relevant API call is made, and the response is fed back to the LLM followed by </API> special token.
* Subsequently, decoding is resumed.

### Experiment –
* Performed under zero-shot setting (no in-context learning examples provided)
* Standard greedy decoding process adopted with slight variation where the API call is generated in the output not only when <API> special token is most likely, but even if it’s among top K most likely tokens
* Evaluation performed on WikiText & held out CCNet dataset to monitor perplexity of different approach variations
* Experimented with 4 tools – calculator, search engine, Q & A, Calendar, Machine translation

### Result takeaways –
* API calls donot seem to be much useful to smaller models (GPT2 family), while the larger models are capable of learning how to make good use of them
* Self-supervised approach enabled the model to learn when & how to use a tool without requiring prompt demonstrating any task specific examples for the same

***Limitations*** – 
* Cannot be used in a chain (where O/P of one tool can be used as I/P for another one)
* Doesn’t let the LLM to use a tool in an interactive way

Paper link - [Toolformer, 2023](https://arxiv.org/pdf/2302.04761)
