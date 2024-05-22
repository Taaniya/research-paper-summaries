## Self-consistency improves CoT reasoning in Language Models, 2023
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


## Chain of thought prompting elicits reasoning in LLMs, 2023
Wei et al.

**Gist** -
* This paper introduces a prompting technique that improves an LLMs’ ability to reason for complex tasks
* This technique is based on including a chain-of-thought demonstrations as exemplars within the prompt to the LLM.
* Experiments show improvements in range of tasks including reasoning, arithmetic, common sense & symbolic reasoning
* Approach evaluated on GSMK benchmark of math word problems

**Approach** –
* The paper defines chain of thought as a series of intermediate reasoning steps in natural language that lead to a final answer
* As part of this approach, the prompt will include triples: input, chain of thought, output
* LLMs are capable of generating chains-of-thought if chain-of-thought reasoning are provided as exemplars in few-shot prompting
* Additionally, this also lets us view thought process of model while solving and helps in debugging how it reached the final result.

**Intuition behind the approach** –
* Analogous to the way human mind solves a complex problem by breaking it down into a sequence of smaller logical problems, solves each of them to reach to the final answer
* The paper aims to incorporate this ability in an LLM while solving complex problems reasonably.

**Few more takeaways** – 
* Model scale matter. CoT works well for LLMs with > 10B parameters and may hurt performance of models smaller than that
* It helps the most when 3 conditions are met –
  - Task is challenging that requires multiple reasoning steps
  - Involves an LLM as the task solver
  - The scaling curve is relatively flat

