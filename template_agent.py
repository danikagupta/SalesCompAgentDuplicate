# src/template_agent.py

from typing import List
from pydantic import BaseModel
from src.create_llm_message import create_llm_message


# Data model for structuring the LLM's response
# Replace PlaceHolder with a new name for this class and new names for param_1 and param_2
class PlaceHolder(BaseModel):
    param_1: str 
    param_2: str  

# Replace 'SubAgentName' and 'subagent_name' with the actual sub-agent name
# When SubAgentNameAgent object is created, it's initialized with a client and a model. 
# The main entry point is the subagent_name_agent method. 
# Note: You will have to add the nodes and edges for this sub-agent node in graph.py

class SubAgentNameAgent:
    """
    A class to handle <insert the purpose of this sub-agent>.
    This agent generates friendly and professional responses to user queries.
    """
    
    def __init__(self, client, model):
        """
        Initialize the SubAgentNameAgent with necessary components.

        Args:
            client: The OpenAI client for API interactions.
            model: The language model to use for generating responses.
        """
        # Initialize the SmallTalkAgent with an OpenAI client and model
        self.client = client
        self.model = model

    def generate_response(self, user_query: str) -> str:
        """
        Generate a response to the user's query using the language model.

        Args:
            user_query (str): The initial message or query from the user.

        Returns:
            str: The generated response from the language model.
        """
        
        # Construct the prompt to guide the language model in generating a response
        subagent_name_prompt = f"""
        You are an expert with deep knowledge of sales compensation. Your job is to <insert the purpose of this 
        sub-agent. Always maintain a friendly, professional, and helpful tone throughout the interaction. 

        The user's query was: "{user_query}"

        Instructions:

        1. abc
        2. def
        3. ghi

        """
        # Create a well-formatted message for LLM by passing the retrieved information above to create_llm_messages
        llm_messages = create_llm_message(subagent_name_prompt)

        # Option 1: Invoke the model with the well-formatted prompt, including SystemMessage, HumanMessage, and AIMessage
        llm_response = self.model.invoke(llm_messages)

        # Option 2: Invoke the model with the well-formatted prompt, including SystemMessage, HumanMessage, and AIMessage
        # Use this option when you would like the output to be in a particular structure i.e., specific parameters
        llm_response = self.model.with_structured_output(PlaceHolder).invoke(llm_messages)

        # Extract the content attribute from the llm_response object 
        subagent_name_response = llm_response

        return subagent_name_response


    def small_talk_agent(self, state: dict) -> dict:
        """
        Process the user's initial message and generate a small talk response.

        Args:
            state (dict): The current state of the conversation, including the initial message.

        Returns:
            dict: An updated state dictionary with the generated response.
        """

                
        # Use this code, in case you have to retrieve relevant document via RAG
        # If you have to then also use this: 
        # full_response = self.generate_response(retrieved_content, state['initialMessage'])
        # And this: def generate_response(self, retrieved_content: List[str], user_query: str) -> str:
        # Retrieve relevant documents based on the user's initial message
        retrieved_content = self.retrieve_documents(state['initialMessage'])


        # Generate a response based on the user's initial message
        full_response = self.generate_response(state['initialMessage'])

        # If you have to put some logic to make a decision, you can enter it here
        
        # Return the updated state with the generated response and the category set to 'smalltalk'
        return {
            "lnode": "small_talk_agent", 
            "responseToUser": full_response,
            "category": "smalltalk"
        }
