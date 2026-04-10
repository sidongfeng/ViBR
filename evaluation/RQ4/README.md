## RQ4: Runtime Overhead

To answer RQ4, we measure the runtime overhead associated with VLM
calls across different replay phases, including interactive region detection, region of
interest selection, attention-driven state comparison, and bug replay
action inference. 

### Setup
1. We execute our approach end-to-end and record the latency and token usage for each OpenAI GPT API call throughout the execution process.

2. For the monetary cost, we compute the total expense based on the official pricing of GPT-4o at the time of evaluation in [here](https://openai.com/api/pricing/). Specifically, OpenAI charges approximately $2.50 per 1 million input tokens and $10.00 per 1 million output tokens for GPT-4o usage via the API.


### Results

<p align="center">
<img src="../../figures/rq4.png" width="75%">
</p>
<p align="center">Table: Runtime overhead associated with VLM calls.</p>


For each GUI recording, ViBR invokes these components once per action scene.
First, ViBR employs Interactive Region Detection using GroundingDINO, which takes an average of
4.17 seconds per scene. Since GroundingDINO is an open-source, locally deployed object detector,
it introduces no monetary cost during execution. Following this step, ViBR performs three GPT-4o-
based reasoning tasks per action scene: (1) Region of Interest Selection (avg. 4.15s), (2) Attention-
Driven State Comparison (avg. 3.93s), and (3) Bug Replay Action Inference (avg. 5.93s). We observe
moderate input-output token usage across these calls, on average 2,362, 2,319, and 3,462 total tokens
(input + output), respectively. 
A typical bug
reproduction involving 10 action scenes would incur a total inference cost of around $0.02.