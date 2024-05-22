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


