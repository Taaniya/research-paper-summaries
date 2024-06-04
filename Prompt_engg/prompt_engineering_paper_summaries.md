## Self-consistency improves CoT reasoning in Language Models, ICLR March 2023
Wang et al

**Gist** - 

•	This paper introduces a novel decoding strategy – self-consistency, to replace naïve greedy decoding used in chain of thought (CoT) prompting

•	Works by sampling multiple diverse reasoning answers & then selecting the most consistent one by marginalizing the sampled answer outputs.

•	Proves improvement in CoT based on evaluation performed on benchmarks involving arithmetic and common reasoning

**Approach** –
* During sampling of multiple answers, assume the answer $a_i$ is from a set of answers, $a_i \forall A$. Given a prompt, and a question, 
self-consistency introduces an additional latent variable $r_i$, which represents the sequence of tokens that are part of the reasoning path in the 
i-th output. These tokens are then followed by the main answer $a_i$ in the end of that path. Generating this $r_i$ is optional and its only purpose is to reach the final answer $a_i$.
* Hence, after decoding multiple $r_i$ & $a_i$ pairs for multiple paths, self-consistency involves applying a marginalization over $r_i$ by taking a majority vote over $a_i$, i.e., argmax.
* Alternatively, we can also weight each ($r_i$, $a_i$) pair by their conditional probability – P( $r_i$, $a_i$ | prompt, question)
This conditional probability can be obtained from log probabilities during generation of each token in the model response.

Paper link - [Self consistency](https://arxiv.org/pdf/2203.11171)

## LLMs can be easily distracted by irrelevant context – Jan 2023
Shi et al

### Gist –
•	This paper investigates the distractibility of large language models, i.e., how the model problem-solving accuracy can be influenced by irrelevant context
•	dataset used - Grade-School Math with Irrelevant Context (GSM-IC) 
•	Share prompting ways to make LLMs robust enough to tackle noise / irrelevant context in the prompt

### Motivation
In real-world situations, where problems usually come with several pieces of contextually related information, which may or may not be relevant to the problems that we want to solve 

### Prompting techniques experimented –
**Few shot** – 
1.	Chain of Thought (CoT) with examplars. Q & A Examples of step by step solving followed by given task.
2.	0-CoT – (Zero shot CoT). Task followed by instruction – let’s think step by step.
3.	Least to Most – LTM
4.	Program prompt
5.	Instructed CoT prompt (feel free to ignore irrelevant info in the question)

In many cases problem examplars had problems described with irrelevant context added in it to make the model understand the problem and the solution in example while ignoring this irrelevant context. Basically making it learn how to tackle noise in the input (which is unavoidable in real world problems)

### Experiment results - 
* Finds that the model performance is dramatically decreased when irrelevant information is included
* Adding more exemplars may also make the prompt less robust since it leads to overfitting
  
### Mitigation –
* Instructed prompt – use natural language to make the model ignore irrelevant info in context. Adding to the prompt an instruction that tells the language model to ignore the irrelevant information
* For few-shot prompts, we find that using exemplars with distractors (i.e., including problems with irrelevant context) consistently outperforms using the original exemplars without distractors across prompting techniques. This way LLM learns from these examples that input can have irrelevant info / distractions and still it has to be robust to understand the expected output
* Decoding with self-consistency 

Paper link - (LLMs can be easily distracted by irrelevant context, 2023)[https://arxiv.org/pdf/2302.00093]

## Chain of thought prompting elicits reasoning in LLMs, 2022
Wei et al.

### Gist -
* This paper introduces a prompting technique that improves an LLMs’ ability to reason for complex tasks
* This technique is based on including a chain-of-thought demonstrations as exemplars within the prompt to the LLM.
* Experiments show improvements in range of tasks including reasoning, arithmetic, common sense & symbolic reasoning
* Approach evaluated on GSMK benchmark of math word problems

### Approach –
* The paper defines chain of thought as a series of intermediate reasoning steps in natural language that lead to a final answer
* As part of this approach, the prompt will include triples: input, chain of thought, output
* LLMs are capable of generating chains-of-thought if chain-of-thought reasoning are provided as exemplars in few-shot prompting
* Additionally, this also lets us view thought process of model while solving and helps in debugging how it reached the final result.

#### Intuition behind the approach –
* Analogous to the way human mind solves a complex problem by breaking it down into a sequence of smaller logical problems, solves each of them to reach to the final answer
* The paper aims to incorporate this ability in an LLM while solving complex problems reasonably.

### Few more takeaways – 
* Model's scale matters. CoT works well for LLMs with > 10B parameters and may hurt performance of models smaller than that
* It helps the most when 3 conditions are met –
  - Task is challenging that requires multiple reasoning steps
  - Involves an LLM as the task solver
  - The scaling curve is relatively flat

Paper link - [Chain of thought prompting](https://openreview.net/pdf?id=_VjQlMeSB_J)
