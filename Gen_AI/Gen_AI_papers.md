## What are tools anyway? A survey from Language Model Perspective (WIP)
Wang et al, 2024

#### Gist 
•	Provide a unified definition of tools as external programs used by LLMs across a broad range of scenarios
•	Perform a systematic review of LM tooling scenarios & approaches
•	Analyze cost efficiency of tooling methods – give practical guidance on when & how one can use tools
•	Offer concrete suggestions for evaluations

#### Limitations of LLMs that tools aim to solve –
•	LLMs struggle to perform tasks that require complex skills – math, complex reasoning 
•	Fundamentally unable to solve tasks that require access to facts or information not included in their training data (e.g., current weather, latest events, current date etc)

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
