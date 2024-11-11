# src/plan_explainer_agent.py

from typing import List
from src.create_llm_message import create_llm_message

# When PlanExplainerAgent object is created, it's initialized with a client, a model, and an index. 
# The main entry point is the plan_explainer_agent method. You can see workflow.add_node for plan_explainer_agent 
# node in graph.py

class PlanExplainerAgent:
    
    def __init__(self, client, model, index):
        """
        Initialize the PlanExplainerAgent with necessary components.
        
        :param client: OpenAI client for API calls
        :param model: Language model for generating responses
        :param index: Pinecone index for document retrieval
        """
        self.client = client
        self.index = index
        self.model = model

    def retrieve_documents(self, query: str) -> List[str]:
        """
        Retrieve relevant documents based on the given query.
        
        :param query: User's query string
        :return: List of relevant document contents
        """
        # Generate an embedding for the query and retrieve relevant documents from Pinecone.
        embedding = self.client.embeddings.create(model="text-embedding-ada-002", input=query).data[0].embedding
        results = self.index.query(vector=embedding, top_k=3, namespace="", include_metadata=True)
        
        retrieved_content = [r['metadata']['text'] for r in results['matches']]
        return retrieved_content

    def generate_response(self, retrieved_content: List[str], user_query: str) -> str:
        """
        Generate a response based on retrieved content and user query.
        
        :param retrieved_content: List of relevant document contents
        :param user_query: Original user query
        :return: Generated response string
        """
        # Construct the prompt to guide the language model in generating a response
        plan_explainer_prompt = f"""
        You are a sales compensation expert with deep knowledge of all sales compensation plan types, plan components, 
        and how they work. The user is inquiring about compensation plan constructs or mechanics. Always maintain a 
        friendly, professional, and helpful tone throughout the interaction.

        Instructions:

        1. Retrieve Relevant Documents: Access the company's Sales Compensation Plans using the provided {retrieved_content}.
        Use the information from these documents to assist the user.
        
        2. Understand Key Terms:

            a) Plan Construct: Three main plan constructs or types exist: Quota Plan, KSO Plan, and Hybrid Plan.
            Quota Plan details include the number of quota buckets, the weight of each bucket, incentive caps, 
            etc. KSO Plan details include the number of Key Sales Objectives (KSOs), their respective weights, incentive 
            caps, etc. Hybrid Plan combines elements of both Quota and KSO plans, including details of each and the percentage of On-Target Incentive (OTI) tied to quotas versus KSOs.
            
            b) Plan Components: Elements such as Base Commission Rate (BCR), Accelerated Commission Rates (ACR1 or 
            ACR2), Kickers (add-on incentives), multi-year downshifts, transition points, multipliers, etc.
            
            c) Plan Mechanics: Describes how different components function within a plan construct. Examples include 
            the presence of kickers, activation points for ACR1 and ACR2, application of multi-year downshifts, etc.
        
        3. Provide an Explanation Using Retrieved Information: Utilize the retrieved documents to explain relevant 
        plan constructs, components, or mechanics that address the user's query.
        
        4. Request Role Clarification if Necessary: If you cannot find specific plan details, kindly ask the user for 
        their role or title (e.g., Account Executive, Account Manager, Solution Consultant, System Engineer, Specialist 
        Sales Rep, etc.). Use this information to search the relevant plan details in the documents again.
        
        5. Leverage Expert Knowledge if Documents Are Insufficient: If, after role clarification, the relevant 
        information is still unavailable, draw upon your extensive knowledge of sales compensation plans, terminologies, 
        policies, and practices typical in a large enterprise software company to assist the user.

        """
        # Create a well-formatted message for LLM by passing the retrieved information above to create_llm_messages
        llm_messages = create_llm_message(plan_explainer_prompt)

        # Invoke the model with the well-formatted prompt, including SystemMessage, HumanMessage, and AIMessage
        llm_response = self.model.invoke(llm_messages)

        # Extract the content attribute from the llm_response object 
        plan_explainer_response = llm_response.content

        return plan_explainer_response

    def plan_explainer_agent(self, state: dict) -> dict:
        """
        Main entry point for plan-related queries.
        
        :param state: Current state dictionary containing user's initial message
        :return: Updated state dictionary with generated response and category
        """
        # Handle plan type, construct, mechanics related queries by retrieving relevant documents and generating 
        # a response.
        
        # Retrieve relevant documents based on the user's initial message
        retrieved_content = self.retrieve_documents(state['initialMessage'])
        
        # Generate a response using the retrieved documents and the user's initial message
        full_response = self.generate_response(retrieved_content, state['initialMessage'])
        
        # Return the updated state with the generated response and the category set to 'policy'
        return {
            "lnode": "plan_explainer_agent", 
            "responseToUser": full_response,
            "category": "planexplainer"
        }
