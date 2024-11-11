Copyright 2024 Jahangir Iqbal

# SALES COMP AGENT INTRODUCTION

Sales Comp Agent is an AI assistant designed to provide front-end support for field sales teams. It reduces the number of support tickets created by sales representatives by providing them with answers to their questions and queries. Trained extensively on sales compensation design, policy, governance, and computations, Sales Comp Agent is equipped to assist with a wide range of inquiries.

Sales Comp Agent consists of 10 sub-agents that work together to understand and answer users' queries accurately and effectively:

1) Small Talk Agent: Answers simple chit-chat.

2) Policy Agent: Addresses all policy-related questions.

3) Commission Agent: Computes estimated commission amounts.

4) Contest Agent: Gathers requirements for sales contests or SPIFFs.

5) Plan Explainer Agent: Explains how any compensation plan works.

6) Feedback Collector Agent: Collects feedback on plan constructs or policies.

7) Insights Agent: Analyzes sales compensation data and provides answers in plain English with supporting data points.

8) Ticket Agent: Initiates human involvement by creating a sales compensation ticket.

9) Exceptions Agent: Gathers requirements for sales compensation exceptions.

10) Clarify Agent: Asks clarifying questions if the user's request is unclear.


# HOW THE CODE WORKS

1) User Input and Initial Classifier:
The user input is passed to the graph.stream method.
The initial_classifier method is invoked since the entry point is set to "classifier".

2) Returning from Initial Classifier:
The initial_classifier method classifies the input and returns a state with the category.
The returned state includes "category": category, where category could be "policy", "commission", etc.

3) Routing in Main Router:
The main_router method receives the state and returns the category, which corresponds to the next node in the state graph.

4) StateGraph Handles Transitions:
The StateGraph framework automatically transitions to the node corresponding to the returned category.
It then calls the method associated with that node.

5) Why do we need both initial_classifier and main_router?
    a) initial_classifier: Focuses solely on classifying the user input. It doesn't decide what to do next.
    b) main_router: Takes the classification result and determines the next state in the graph. 

This decouples the classification logic from the routing logic.

# HOW TO UPLOAD DOCUMENTS FOR RETRIEVAL AUGMENTED GENERATION (RAG) - For System Administrators

1) RAG script is in the file called rag.py. This is only for system administrator and not for users.

2) When you run the script, it will upload the PDF or Markdown file. When you click on "Process file", it will convert it to text, generate embeddings, and upload those embeddings to Pinecone.

3) Now, you can go to Pinecone and validate that new vectors have been added.

4) You're all set! 

