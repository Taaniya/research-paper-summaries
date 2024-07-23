## MAGPIE: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing, June 2024
Zhangchen Xu et al, 2024

#### Gist –
* This paper introduces MAGPIE, a large scale approach to self-synthesize alignment data for LLMs without requiring human intervention or external APIs. It uses open source aligned LLMs to generate instruction data instances
* Motivation is to tackle current limitation of alignment dataset generation that requires
* Huge man efforts to generate and curate instruction data
* Alternative methods that use LLMs to synthesize the dataset, though reducing human efforts, still require careful prompt engineering & selection of initial seed questions
* Compares the approach over alignment benchmark by finetuning llama 3 8B base on MAGPIE datasets & existing public instruction datasets and comparing with current SOA instruction models
* Instruct models used to generate datasets – Llama-3-8B Instruct and Llama-3-70B Instruct models

#### Approach -
* Each instance of the instruction dataset consists of instruction & response pairs, where the instruction part specifies the role of the instruction provider (user) and the follower, along with the instruction
* Given an open sourced LLM, MAGPIE creates a query template in a pre-defined format of the instruction describing the role of instruction provider without containing the instruction itself.
* Passing this query to the LLM, results in generation of a user instruction by the LLM in an automated form, as if to complete the input query, due to its autoregressive nature.
* Passing this query to the LLM multiple times results in a set of many diverse instructions
* Subsequently, each instruction is fed to the LLM to generate their corresponding responses
* This method generated 4 Million instruction & response pairs, out of which 300 K high quality instances have been selected as final dataset

#### Dataset details
* This paper releases 2 instruction datasets – MAGPIE-Air & MAGPIE-Pro constructed using Llama-3-8B-Instruct & Llama-3-70B-Instruct resp.
* Also generated 2 multi-turn instruction datasets with sequences of multi-turn instruction & responses

Paper link - [MAGPIE](https://arxiv.org/pdf/2406.08464)

Git repo - https://magpie-align.github.io/


