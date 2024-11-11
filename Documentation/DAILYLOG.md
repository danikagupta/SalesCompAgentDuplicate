DAILY PROGRESS LOG:

11/7/2024:
- Updated the prompt for commission agent. Working well with no formatting issues.
- The current prompt calculates the commission using BCR and does not take into account current quota attainment
- If the user provides that additional input then it performs the calculations correctly
- This is something I need to think about whether I'd like the assistant to ask for this information or just clarify that these calculations do not take into account any acceleration.

11/6/2024:
- Fixed the formatting issue for commission agent responses. This required a tricky solution because chat history was not formatting correctly.

11/2/2024:
- Finalized the code for contest_agent. The agent workflow is working well.

11/1/24: 
- I updated the prompt for "plan_explainer_agent.py". 
- Updated the Apachee license info.
- Added feedback_collector_agent to graph.py

10/31/24: 
- I worked on "feedback_collector_agent.py". 
- I've updated the code in this file. 
- I also worked on writing a detailed prompt. I used ChatGPT to refine my prompt. I think I should try to optimize all prompts for other sub-agents as well. However, I think I should do that carefully and do some sort of A/B testing to optimize the prompts.