# Papers
1. [BOLAA – Benchmarking and orchestrating LLM autonomous agents, ICLR 2024](#bolaa--benchmarking-and-orchestrating-llm-autonomous-agents) 
2. [ChatDev - Communicative Agents for Software Development, ACL 2024](#chatdev---communicative-agents-for-software-development)


## BOLAA – Benchmarking and orchestrating LLM autonomous agents
Liu et al, Salesforce, ICLR 2024

* Paper link - https://openreview.net/pdf?id=BUa5ekiHlQ
* Git repo - https://github.com/salesforce/BOLAA


**Gist –** 
* This paper performs comparison different LLM based autonomous agents (LAA) in terms of agent architectures and LLM models used within them
* Explores new areas –
 * Compatibility of different LLM models with task complexities and agent architectures
 * Strategic orchestration of multiple agents with increasing task complexity
* Proposes a new strategy to orchestrate multiple LAAs with architecture BOLAA to orchestrate multiple agents and also identify the challenges of designing this architecture for environments with compounding actions.
* This paper studies following 2 core modules of BOLAA on the basis of LLM based and heuristic based methods -
 * Agent selection to select most relevant agent
 * Communication module to communicate between multiple agents

**LAA architectures-**
* This paper defines and compares different agent architectures (LAAs) –
 * Zeroshot LAA – This basically extends LLM to be an action executor and is a minimum LAA architecture
 * ZeroshotThink LAA – Extends Zeroshot LAA with an additional think flow using Chain-Of-Thought (CoT) reasoning ability
 * ReAct LAA – Extends ZeroshotThink LAA by providing few shot examples to enhance the action generation and environment interaction ability of the LAA
 * PlanAct LAA – Zeroshot LAA and facilitates planning ability of LAA by having a planning step and includes few shot examples in the prompt
 * PlanReAct LAA extends PlanAct LAA with additional self-think flow, which also enables the CoT ability.
 * BOLAA – To tackle challenges faced in above architectures
   * Context length constraints
   * ICL – in context learning
   * Generalization ability

**BOLAA architecture-**
* Has following 2 main modules -
  * Labor agents pool that manager multiple LAA each focusing on specialized tasks
  * Controller - The controller is devised to selectively call LAAs from agents pool.
* Controller has 2 core modules that involve performing agent selection and communication between multiple labor agents
* **Agent select module –**
  * Heuristic based approach works by defining rules to select relevant agent while LLM based method present the agent selection task as the action generation process for the LLM.
  * This way the controller plays the role of an orchestrating agent itself whose action is the select the optimal labor agent.

* **Communication module –**
* Communication is core to agent orchestration. The controller constructs the message after selecting the LAA in previous step And builds the communication.
* Heuristic based methods generates message using pre-defined template while LLM based method involves the LLM to generate the communication using a prompt that includes – label agent responses, their details, task instructions, demo examples, execution history etc. After getting the response from labor LAA, the controller parses it into an executable action and interacts with the environment.

### Experiment & result analysis -
**Evaluation -**
* Benchmark datasets – HotPotQA, WebShop
* Evaluation metrics –
  * Reward score defined in each environment as per task for e.g., custom ratio metric, Recall and F1 score

**Takeaways from experimental results -**
* BOLAA performs the best compared with the other LAA architectures, especially when built on
the high performing LLMs.
* Selecting the appropriate LAA yields quality communication and stabilizes action generation
* Superiority of BOLAA indicates that orchestrating multiple smaller sized LAA is a better choice if the computing resources are limited.
* The optimal architecture of agents should be aligned with both tasks and the associated LLM model
* Pairing LLM model with optimal LLM architecture is crucial
* Increasing complexity of tasks may require the orchestration of multiple agents
* Designing specialist agents to collaborate on resolving complex task should be equally important as training a large LLM with high generalization ability.
* Increasing context length of LAA many not alone improve their performance. For e.g., they could suffer more from issues like hallucinations when LAA run for more steps.
* A powerful LLM (E.g., Open AI based LLMs) is able to generalize well under ZeroShot LAA architecture. However, for other less powerful LLMs, few shot prompts are necessary for LAAs.
* Planning flow of LAA hinders performance in the knowledge reasoning tasks and environments.




## ChatDev - Communicative Agents for Software Development

Qian et al, ACL, 2024

**Gist-**
* This paper introduces Chatdev, a system with communicative agents for software development
* Each agent is driven by LLM and is guided by what to communicate and how to communicate
* Demonstrates how linguistic communication facilitates multi-agent collaboration
* Introduces Chat chain to divide each phase into smaller subtasks and communicative de-hallucination to ensure genuine response generation during communication

### Chat chain –
* Adopting waterfall model, software development is broken down to sequential phases – design, coding & testing, where each phase is further divided into subtasks.
* In each subtask, 2 agents with their specialized roles perform function of an instructor and an assistant.
* The instructor agent initiates instructions, instructing the discourse toward the completion of the subtask, while the assistant agent adheres to these instructions and responds with appropriate solutions.
* They engage in a multi-turn dialogue working cooperatively until they achieve consensus, extracting solutions
that can range from the text (e.g., defining a software function point) to code (e.g., creating the initial version of source code), ultimately leading to the completion of the subtask
Subsequently, the solution from previous tasks serve as bridges to the next phase

### Agentic workflow -
* Follows agentic workflow, where prompt engineering only takes place at the start of each subtask round.
* As the communication phase begins, the instructor and assistant communicate with each other in an automated loop, continuing this exchange until the task concludes.
* Some challenges faced here – Role flipping, instruction repeating, fake replies etc, resulting in failure to advance progress of productive communications and hinders achievement of meaningful solutions
* This paper employs inception prompting mechanism to tackle this for initiating, sustaining, and concluding agents’ communication to guarantee a robust and efficient workflow.


### Memory –
* To tackle the issue arising from constraint of context length in LLM while maintaining communication history among all agents and phases, agent’s memories are segmented into –
  * Long term - preserve contextual awareness across different phases
  * Short term - sustain continuity of dialogue in a single phase

### Communication dehallucination –
* Coding hallucinations usually occur when the assistant LAA struggles to precisely follow instructions. To tackle this, dehallucination encourages the assistant to actively seek more detailed suggestions from instructor before delivering a formal response.

