# src/feedback_collector_agent.py

from typing import List
from src.create_llm_message import create_llm_message

# When FeedbackCollectorAgent object is created, it's initialized with a client, a model, and an index. 
# The main entry point is the feedback_collector_agent method. You can see workflow.add_node for feedback_collector_agent 
# node in graph.py

class FeedbackCollectorAgent:
    
    def __init__(self, model):
        """
        Initialize the FeedbackCollectorAgent with necessary components.
        
        :param model: Language model for generating responses
        """
        self.model = model


    def generate_response(self, user_query: str) -> str:
        """
        Generate a response based on user query.
        
        :param user_query: Original user query
        :return: Generated response string
        """
        # Construct the prompt to guide the language model in generating a response
        feedback_collector_prompt = f"""
        You are a sales compensation expert with deep knowledge of all sales compensation plans, policies, SPIFs 
        (Sales Performance Incentive Funds), and sales contests. The user is providing feedback on what is working 
        and what is not working in their current compensation plan, policy, SPIF, or sales contest. Always maintain 
        a friendly, professional, and helpful tone throughout the interaction.

        Instructions:

        1. Identify the Specific Area of Feedback: Determine whether the user's feedback pertains to a sales compensation 
        plan, policy, SPIF, or sales contest.
        
        2. Seek Clarification: If the feedback is not specific enough, ask a well-articulated question in plain English 
        to deeply understand the cause of dissatisfaction. Ensure the question is pointed and invites detailed information.
        
        3. Request a Specific Example: If the user hasn't shared an example, politely ask them to provide a specific 
        scenario, use case, or example that illustrates their feedback.
        
        4. Acknowledge and Summarize: Acknowledge that you believe you have understood the issue. Rephrase the user's 
        feedback and example in a clear, concise summary to confirm your understanding.
        
        5. Confirm Accuracy: Ask the user to confirm that your summary accurately and completely captures their feedback.
        Example: "Have I captured your concerns correctly?"
        
        6. Address Incomplete or Inaccurate Summaries: If the user indicates that the summary is inaccurate or incomplete, 
        incorporate the missing information. Rewrite the summary and ask for confirmation again.
        
        7. Document the Feedback: Once the user agrees that the feedback is accurately captured, document it in a 
        "Sales Compensation Feedback" report. Ensure the document is well-formatted and professional.
        
        8. Express Gratitude and Next Steps: Thank the user for providing their feedback. Inform them that you have 
        documented their feedback and will share it with the Sales Compensation team.
            
        """
        # Create a well-formatted message for LLM by passing the retrieved information above to create_llm_messages
        llm_messages = create_llm_message(feedback_collector_prompt)

        # Invoke the model with the well-formatted prompt, including SystemMessage, HumanMessage, and AIMessage
        llm_response = self.model.invoke(llm_messages)

        # Extract the content attribute from the llm_response object 
        feedback_collector_response = llm_response.content

        return feedback_collector_response

    def feedback_collector_agent(self, state: dict) -> dict:
        """
        Main entry point for feedback related queries.
        
        :param state: Current state dictionary containing user's initial message
        :return: Updated state dictionary with generated response and category
        """
        
        # Generate a response using the retrieved documents and the user's initial message
        full_response = self.generate_response(state['initialMessage'])
        
        # Return the updated state with the generated response and the category set to 'policy'
        return {
            "lnode": "feedback_collector_agent", 
            "responseToUser": full_response,
            "category": "feedbackcollector"
        }
