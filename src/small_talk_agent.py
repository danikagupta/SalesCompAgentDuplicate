# src/small_talk_agent.py

from typing import List
from src.create_llm_message import create_llm_message

# When SmallTalkAgent object is created, it's initialized with a client and a model. 
# The main entry point is the small_talk_agent method. You can see workflow.add_node for small_talk_agent node in graph.py

class SmallTalkAgent:
    """
    A class to handle small talk interactions in a sales compensation context.
    This agent generates friendly and professional responses to user queries.
    """
    
    def __init__(self, client, model):
        """
        Initialize the SmallTalkAgent with necessary components.

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
        small_talk_prompt = f"""
        You are an expert with deep knowledge of sales compensation. Your job is to comprehend the message from 
        the user even if it lacks specific keywords, always maintain a friendly, professional, and helpful tone. 
        If a user greets you, greet them back by mirroring user's tone and verbosity, and offer assitance. 

        User's message: {user_query}

        Please respond to the user's message:
        """
        # Create a well-formatted message for LLM by passing the retrieved information above to create_llm_messages
        llm_messages = create_llm_message(small_talk_prompt)

        # Invoke the model with the well-formatted prompt, including SystemMessage, HumanMessage, and AIMessage
        llm_response = self.model.invoke(llm_messages)

        # Extract the content attribute from the llm_response object 
        small_talk_response = llm_response.content

        return small_talk_response


    def small_talk_agent(self, state: dict) -> dict:
        """
        Process the user's initial message and generate a small talk response.

        Args:
            state (dict): The current state of the conversation, including the initial message.

        Returns:
            dict: An updated state dictionary with the generated response.
        """
        # Generate a response based on the user's initial message
        full_response = self.generate_response(state['initialMessage'])
        
        # Return the updated state with the generated response and the category set to 'smalltalk'
        return {
            "lnode": "small_talk_agent", 
            "responseToUser": full_response,
            "category": "smalltalk"
        }
